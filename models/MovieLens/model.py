import jax
from jax import numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import scale

def model(X=None, y=None, group_id=None, **kwargs):
    num_groups = kwargs['num_groups']
    num_total_sample = kwargs['N']
    D = X.shape[1] if len(X.shape) == 2 else X.shape[0] # With vmap, it's very likely X is 1-D
    N = len(X)
    mu = numpyro.sample(
        'mu',
        dist.Normal(loc=jnp.zeros((D, 5)), scale=jnp.ones((D, 5)))
    )
    log_scale = numpyro.sample(
        'scale',
        dist.Normal(loc=jnp.zeros((D, 5)), scale=jnp.ones((D, 5)))
    )
    group_weight = numpyro.sample(
        'z',
        dist.Normal(
            loc=mu + jnp.zeros((num_groups, D, 5)),
            scale=jnp.exp(log_scale) * jnp.ones((num_groups, D, 5))
        )
    ) # (M, 18, 5)
    z_group = group_weight[group_id] # (BatchSize, 18, 5)
    if len(z_group.shape) == 1:
        logits = X @ z_group
    else:
        logits = jax.vmap(jnp.dot)(X, z_group)
    with scale(scale=num_total_sample / N):
        numpyro.sample("obs", dist.Categorical(logits=logits),
                       obs=y
        )