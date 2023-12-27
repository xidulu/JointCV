import jax
from jax import numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import scale

def model(winner_ids=None, loser_ids=None, **kwargs):
    num_player = kwargs['num_player']
    num_total_sample = kwargs['N']
    N = len(winner_ids)
    player_sd = numpyro.sample(
        "player_sd",
        dist.HalfNormal(1.0)
    )
    player_skills = numpyro.sample(
        'player_skills',
        dist.Normal(loc=jnp.zeros(num_player), scale=player_sd * jnp.ones(num_player))
        # dist.Normal(loc=jnp.zeros(num_player), scale=jnp.ones(num_player)) # For debug only
    )
    logit_skills = player_skills[winner_ids] - player_skills[loser_ids]
    with scale(scale=num_total_sample / N):
        numpyro.sample("obs", dist.Bernoulli(logits=logit_skills),
                       obs=jnp.ones(winner_ids.shape[0])
        )