import jax
import jax.numpy as jnp

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.maximum(outputs, 0)
  return outputs

def loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))


loss_jit = jax.jit(loss)
gradfun = jax.jit(jax.grad(loss))

def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = jax.random.normal(k2, (n_out,))
    return W, b

def init_model(key, layer_sizes, batch_size):
    key, *keys = jax.random.split(key, len(layer_sizes))
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

    key, *keys = jax.random.split(key, 3)
    inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))

    return params, (inputs, targets)

layer_sizes = [784, 8192, 8192, 8192, 10]
batch_size = 8192

params, batch = init_model(jax.random.PRNGKey(0), layer_sizes, batch_size)

sharding = PositionalSharding(jax.devices()).reshape(8, 1)
batch = jax.device_put(batch, sharding)
params = jax.device_put(params, sharding.replicate())

loss_jit(params, batch)

step_size = 1e-5

for _ in range(30):
  grads = gradfun(params, batch)
  params = [(W - step_size * dW, b - step_size * db)
            for (W, b), (dW, db) in zip(params, grads)]

print(loss_jit(params, batch))

# 4-way batch data parallelism and 2-way model tensor parallelism

sharding = sharding.reshape(4, 2)
batch = jax.device_put(batch, sharding.replicate(1))
jax.debug.visualize_array_sharding(batch[0])
jax.debug.visualize_array_sharding(batch[1])

(W1, b1), (W2, b2), (W3, b3), (W4, b4) = params

W1 = jax.device_put(W1, sharding.replicate())
b1 = jax.device_put(b1, sharding.replicate())

W2 = jax.device_put(W2, sharding.replicate(0))
b2 = jax.device_put(b2, sharding.replicate(0))

W3 = jax.device_put(W3, sharding.replicate(0).T)
b3 = jax.device_put(b3, sharding.replicate())

W4 = jax.device_put(W4, sharding.replicate())
b4 = jax.device_put(b4, sharding.replicate())

params = (W1, b1), (W2, b2), (W3, b3), (W4, b4)

jax.debug.visualize_array_sharding(W2)

import pdb; pdb.set_trace()

traverse_util.unflatten_dict(train_state.params, sep='/')