import os
from importlib.machinery import SourceFileLoader
import os
import numpy as np
from jax.flatten_util import ravel_pytree
import jax
from numpyro.infer.util import initialize_model


def _inference_kit_generator(numpyro_model_func):
    key = jax.random.PRNGKey(0) # The key doesn't matter here
    def _inner(*args, **kwargs):
        init_funcs = initialize_model(
            key,
            numpyro_model_func, 
            model_args=args,
            model_kwargs=kwargs
        )
        param_template = init_funcs.param_info.z
        potential_func = init_funcs.potential_fn
        transform_func = init_funcs.postprocess_fn
        flatten_func = lambda p: ravel_pytree(p)[0]
        unflatten_func = ravel_pytree(param_template)[1]
        return {
            'param_template': param_template,
            'potential_func': potential_func,
            'transform_func': transform_func,
            'flatten_func': flatten_func,
            'unflatten_func': unflatten_func
        }
    return _inner


class Posterior:
    def __init__(self, model_dir, data_name=None):
        '''
        model_dir: a string
        dataset: a string
        '''
        if not os.path.exists(model_dir):
            raise NotImplementedError(
                f'Directory {model_dir} not found'
            )
        self.model_dir = model_dir
        self.data_name = data_name
        if not self.data_name:
            self.name = f"{'__'.join(model_dir.split('/')[-2:])}"
        else:
            self.name = f"{'__'.join(os.path.join(model_dir, data_name).split('/')[-2:])}"

    @property
    def model_name(self):
        return self.model_dir.split('/')[-1]

    def numpyro(self):
        file_name = os.path.join(self.model_dir, 'model.py')
        if not os.path.exists(file_name):
            raise NotImplementedError(
                    f'NumPyro model file for {self.model_name} not found under {self.model_dir}'
            )
        return getattr(
                SourceFileLoader('', file_name).load_module(), "model"
        )
    
    def numpy(self):
        return _inference_kit_generator(self.numpyro())

    def data(self):
        if not self.data_name:
            return {}
        data_dir = os.path.join(self.model_dir, self.data_name)
        return np.load(data_dir + '.npz')