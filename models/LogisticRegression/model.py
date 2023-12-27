import jax
from jax import numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import scale

def model(X=None, y=None, **kwargs):
    num_total_sample = kwargs['N']
    D = X.shape[1] if len(X.shape) == 2 else X.shape[0]
    N = X.shape[0] if len(X.shape) == 2 else 1
    w = numpyro.sample('w', dist.Normal(jnp.zeros((D,)), jnp.ones((D,))))
    logit = X @ w
    with scale(scale=num_total_sample / N):
        numpyro.sample("obs", dist.Bernoulli(logits=logit), obs=y)
