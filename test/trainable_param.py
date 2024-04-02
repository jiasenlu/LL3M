import optax
import jax
import jax.numpy as jnp

def map_nested_fn(fn):
  '''Recursively apply `fn` to the key-value pairs of a nested dict'''
  def map_fn(nested_dict):
    return {k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()}
  return map_fn

params = {'linear_1': {'w': jnp.zeros((5, 6)), 'b': jnp.zeros(5)},
          'linear_2': {'w': jnp.zeros((6, 1)), 'b': jnp.zeros(1)}}
gradients = jax.tree_util.tree_map(jnp.ones_like, params)  # dummy gradients


label_fn = map_nested_fn(lambda k, _: k)

tx = optax.multi_transform({'w': optax.adam(1.0), 'b': optax.sgd(1.0)},
                           label_fn)
state = tx.init(params)
updates, new_state = tx.update(gradients, state, params)
new_params = optax.apply_updates(params, updates)