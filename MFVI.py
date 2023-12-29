import optax
import jax
import numpy as np
import jax.numpy as jnp
from jax import vmap, grad, value_and_grad, jvp
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from numpyro.infer.util import potential_energy
from jax.random import PRNGKey, split
from jax.tree_util import tree_map
from functools import partial
from models import Posterior

def split_given_size(a, size):
    return jnp.split(a, jnp.arange(size, len(a), size))

def generate_batch_index(key, N, batch_size):
    idx = jnp.arange(N)
    shuffled_idx = split_given_size(jax.random.permutation(key, idx), batch_size)
    return shuffled_idx

def get_optimizer(OPT, step_size):
    assert OPT in ['adam', 'sgd']
    if OPT == 'sgd':
        return optax.sgd(step_size, momentum=0.9)
    return optax.adam(step_size)

class MFVI_with_subsampling:
    def __init__(self, model_dir, dataset, observed_vars=[]):
        """
        model_dir: Directory to the model file
        dataset: Name of the dataset
        observed_vars: A list of observed variable names
        """
        model = Posterior(model_dir, dataset)
        kit_generator = model.numpy()
        self.model_func = model.numpyro()
        self.dataset = {}
        for k, v in model.data().items():
            if isinstance(v, np.ndarray):
                self.dataset[k] = jnp.array(v)
            else:
                self.dataset[k] = v
        self.observed_vars = observed_vars
        kit = kit_generator(**self.dataset)
        self.flattened_param_template = ravel_pytree(kit['param_template'])[0]
        self.unflatten_func = kit['unflatten_func']

    def get_loss_eps_grad(self, key, params, idx, local_reparam=True):
        loc = params['loc']
        if local_reparam:
            eps = jax.random.normal(key, shape=(len(idx[0]),) + loc.shape)
            loss, grads = vmap(value_and_grad(self.loss_func), (None, 0, 0))(
                params, idx, eps
            )
        else:
            eps = jax.random.normal(key, shape=loc.shape)
            loss, grads = vmap(value_and_grad(self.loss_func), (None, None, 0))(
                params, eps, idx)
        return loss, grads, eps

    def get_subsampled_dataset(self, idx):
        new_dict = {}
        for k, v in self.dataset.items():
            if k in self.observed_vars:
                new_dict[k] = v[idx]
            else:
                new_dict[k] = v
        return new_dict

    def get_log_p_func(self, idx):
        kwargs = self.get_subsampled_dataset(idx)
        @jax.jit
        def _inner(params):
            return -potential_energy(
                self.model_func,
                model_args = [],
                model_kwargs = kwargs,
                params=params
            )
        return _inner

    def elbo(self, sample, log_q_func, log_p_func):
        log_p = log_p_func(self.unflatten_func(sample))
        log_q = log_q_func(sample).sum()
        return log_q - log_p

    @partial(jax.jit, static_argnums=(0,))
    def loss_func(self, params, eps, idx):
        loc, log_scale = params["loc"], params["log_scale"]
        log_q_func = dist.Normal(loc, jnp.exp(log_scale)).log_prob
        log_p_func = self.get_log_p_func(idx)
        z = loc + jnp.exp(log_scale) * eps
        return self.elbo(z, log_q_func, log_p_func)

    def eval_fulldataset_loss(self, key, params):
        loc = params['loc']
        key, _key = split(key)
        shuffled_idx = generate_batch_index(_key, self.dataset['N'], 5000)
        losses = []
        for idx in shuffled_idx:
            key, _key = split(key)
            eps = jax.random.normal(_key, shape=(500,) + loc.shape)
            losses.append(vmap(self.loss_func, (None, 0, None))(
                params, eps, idx).mean()
            )
        return np.mean(losses)

    def run(self, step_size=1e-3, seed=1, opt='adam', batch_size=5, num_iters=10000,
                    init_sigma=0.001, local_reparam=False, log_frequency=100):
        raise NotImplementedError


class MFVI_with_subsampling_naive(MFVI_with_subsampling):
    '''
    Using naive doubly stochastic gradient
    '''
    def run(self, step_size=1e-3, seed=1, opt='adam', batch_size=5, num_iters=10000,
                    init_sigma=0.001, local_reparam=False, log_frequency=100):
        key = PRNGKey(seed)
        key, _key = split(key)
        loc, log_scale = (
            jax.random.normal(_key, self.flattened_param_template.shape) / 100,
            jnp.ones_like(self.flattened_param_template) * init_sigma,
        )
        params = {"loc": loc, "log_scale": log_scale}
        losses = []
        grad_norms = []
        optimizer = get_optimizer(opt, step_size)
        opt_state = optimizer.init(params)
        iter_counter = 0
        while iter_counter <= num_iters:
            key, _key = split(key)
            shuffled_idx = generate_batch_index(_key, self.dataset['N'], batch_size)
            for idx in shuffled_idx:
                key, _key = split(key)
                loss, grads, eps = self.get_loss_eps_grad(_key, params, idx, local_reparam)
                grad_norms.append(
                    (grads['loc'] ** 2).mean()
                )
                grads = tree_map(lambda g: g.mean(0), grads)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                if iter_counter % log_frequency == 0:
                    key, _key = split(key)
                    losses.append(self.eval_fulldataset_loss(_key, params))
                iter_counter += 1 
        return params, losses, np.array(grad_norms)


class MFVI_with_subsampling_jointCV(MFVI_with_subsampling):
    """
    Using the SVRG version of joint control variate
    """
    @partial(jax.jit, static_argnums=(0,))
    def get_hessian_vector_product(self, params, idx, eps):
        """
        Compute the hessian-vector product (Eq.63):
        Hessian(-log p(dataset[idx]; theta=mu)) @ (eps * sigma) 
        """
        loc, log_scale = params["loc"], params["log_scale"]
        log_q_func = lambda x: jnp.zeros_like(x) # Not using control variate for the entropy
        log_p_func = self.get_log_p_func(idx)
        scale_noise_product = eps * jnp.exp(log_scale)
        elbo_func = partial(self.elbo, log_q_func=log_q_func, log_p_func=log_p_func)
        hvp = jvp(grad(elbo_func), (loc,), (scale_noise_product,))[1]
        return hvp

    @partial(jax.jit, static_argnums=(0,))
    def get_sample_grad(self, params, idx):
        """
        Gradient of -log p(dataset[idx]; theta) with respect to theta (Eq.63)
        """
        loc, log_scale = params["loc"], params["log_scale"]
        log_p_func = self.get_log_p_func(idx)
        z = loc
        return grad(lambda z: -log_p_func(self.unflatten_func(z)))(z)

    def run(self, step_size=1e-3, seed=1, opt='adam', batch_size=5, num_iters=10000,
                    init_sigma=0.001, local_reparam=False, log_frequency=100,
                    inner_loop_size=None):
        key = PRNGKey(seed)
        key, _key = split(key)
        loc, log_scale = (
            jax.random.normal(_key, self.flattened_param_template.shape) / 100,
            jnp.ones_like(self.flattened_param_template) * init_sigma,
        )
        params = {"loc": loc, "log_scale": log_scale}
        losses = []
        grad_norms = []
        optimizer = get_optimizer(opt, step_size)
        opt_state = optimizer.init(params)
        iter_counter = 0
        # Dual CV specific parameters
        if inner_loop_size:
            inner_loop_size = inner_loop_size
        else:
            inner_loop_size = self.dataset['N'] // batch_size # Update the cache every epoch
        grad_mean = 0.0
        # Main training loop
        while iter_counter <= num_iters:
            key, _key = split(key)
            shuffled_idx = generate_batch_index(_key, self.dataset['N'], batch_size)
            for idx in shuffled_idx:
                if iter_counter % inner_loop_size == 0:
                    old_params = params
                    grad_mean = vmap(self.get_sample_grad, (None, 0))(
                        old_params, jnp.arange(self.dataset['N'])
                    ).mean(0) # Eq. 64
                key, _key = split(key)
                loss, grads, eps = self.get_loss_eps_grad(_key, params, idx, local_reparam)
                grad_norms.append(
                    (grads['loc'] ** 2).mean()
                )
                eps_vmap_flag = 0 if local_reparam else None
                cv_term_0 = vmap(self.get_sample_grad, (None, 0))(
                    old_params, idx
                )
                cv_term_1 = vmap(self.get_hessian_vector_product, (None, 0, eps_vmap_flag))(
                    old_params, idx, eps
                )
                grads['loc'] = grads['loc'] - (cv_term_0 + cv_term_1) + grad_mean
                grad_norms.append((grads['loc'] ** 2).mean())
                grads = tree_map(lambda g: g.mean(0), grads) 
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                if iter_counter % log_frequency == 0:
                    key, _key = split(key)
                    losses.append(self.eval_fulldataset_loss(_key, params))
                iter_counter += 1
        return params, losses, np.array(grad_norms)