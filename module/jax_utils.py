'''
Adapt from https://github.com/young-geng/EasyLM/blob/main/EasyLM/jax_utils.py
'''

import abc
import os
import math
from typing import Any, Mapping, MutableMapping, Tuple, Union, NamedTuple, Sequence, Optional, Set, Callable
from functools import partial
import functools
import cached_property
import re
import dataclasses
import random
import collections

from clu.data import DatasetIterator
from jax.experimental.multihost_utils import host_local_array_to_global_array, \
    global_array_to_host_local_array
from ml_collections import ConfigDict
from ml_collections.config_dict.config_dict import placeholder
import seqio
import tensorflow as tf
import clu
import threading
import typing_extensions
import time

import flax
import jax
import jax.numpy as jnp
from flax.training import common_utils
from jax.sharding import PartitionSpec as PS
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from jax.lax import with_sharding_constraint as _with_sharding_constraint
from jax.experimental.pjit import pjit
from jax.interpreters import pxla
import numpy as np
from transformers import FlaxLogitsWarper
from jax.experimental import multihost_utils

from module import metrics as metrics_lib
from clu import metric_writers
from clu import asynclib

from module.utils import pop_metadata

PyTree = Any

@jax.jit
def _merge_metrics(a, b):
  return jax.tree_util.tree_map(
      lambda a, b: a.merge(b), a, b, is_leaf=metrics_lib.is_metric_obj)


def _time() -> float:
  """Indirection to `time.time` for mocking."""
  return time.time()


# Merges two metrics pytrees (mapping of metric_name (str) to clu.Metric object)
def merge_metrics(a, b):
  if a is None:
    return b
  if b is None:
    return a
  a, b = jax.tree_util.tree_map(get_local_data, (a, b))
  return _merge_metrics(a, b)


def get_local_data(x):
  """Get local buffer for input data."""
  if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer):
    return x.addressable_data(0)
  else:
    return x
  
def multihost_assert_equal(input_tree, fail_message: str = ''):
  """Verifies that all the hosts have the same tree of values."""
  # Internal mock TPU handling
  multihost_utils.assert_equal(input_tree, fail_message)

def _hashed_index(x) -> int:
  # This works for both `pjit`/`xmap` indices and `pmap` indices (which might
  # have an integer instead of a slice).
  assert all(v.step is None for v in x if isinstance(v, slice))
  return hash(
      tuple((v.start, v.stop) if isinstance(v, slice) else v for v in x)
  )

def _get_index_mappings(device_to_idxs):
  """Get device and host to index set mappings for GDA construction."""
  host_to_idxs = collections.defaultdict(list)
  idx_to_devices = collections.defaultdict(list)
  for d, idx in device_to_idxs.items():
    hashed_idx = _hashed_index(idx)
    # Only need one copy of each idx, since they are unique. Need to maintain
    # original ordering though.
    if hashed_idx not in host_to_idxs[d.process_index]:
      host_to_idxs[d.process_index].append(hashed_idx)
    # Index may correspond to multiple devices.
    idx_to_devices[hashed_idx].append(d)

  assert jax.process_index() in host_to_idxs
  for h1, idxs1 in host_to_idxs.items():
    for idx in idxs1:
      assert idx in idx_to_devices
    for h2, idxs2 in host_to_idxs.items():
      if h1 == h2:
        continue
      assert not (set(idxs1) & set(idxs2)) or set(idxs1) == set(idxs2)

  return host_to_idxs, idx_to_devices

def _create_sharded_array(
    partitioner: Any,
    global_shapes: PyTree,
    host_arrays: PyTree,
) -> PyTree:
  """Create jax.Array from input arrays.

  Example:

  Consider a case where the global input array has length 128. The global mesh
  specifies that the data dimension be sharded into 8 shards. This means we want
  shards of length 16. The data_layout, defined by the partitioner object,
  specifies that the data should be divided into two shards, one per host. Each
  host will have a local slice of the data (length 64).

  In this function, we will divide the local array into 4 shards of length 16.
  Each of these will be placed onto a separate device. If the sharding had
  specified only 4 global shards instead of 8, we would have divided our local
  array into only 2 shards. In this case, the first shard would be placed on the
  first two devices (replicated) and the second on the following two devices.

  Args:
    partitioner: Partitioner object containing mesh and mesh_axes
    global_shapes: PyTree matching host_arrays specifying global shape of each
      array.
    host_arrays: PyTree of LOCAL arrays (not global) that should be converted to
      jax.Array.

  Returns:
    PyTree matching host_arrays of jax.Array.
  """
  global_mesh = partitioner.mesh
  axes = partitioner.data_partition_spec
  
  local_devices = global_mesh.local_devices
  local_device_count = jax.local_device_count()

  # Global input array is already split into per-host shards.
  def _put_to_devices(x, global_shape):
    # Mapping of device to index slice from *global* array.
    device_to_idxs = jax.sharding.NamedSharding(
        global_mesh, axes
    ).devices_indices_map(global_shape)
    # Mapping of host to a set of unique index slices for that host.
    # Mapping of index slice to a list of devices onto which the slice should be
    # placed.
    host_to_idxs, idx_to_devices = _get_index_mappings(device_to_idxs)

    shard_length = jax.sharding.NamedSharding(global_mesh, axes).shard_shape(
        global_shape
    )[0]
    num_shards = len(x) // shard_length
    try:
      local_array_shards = np.split(x, num_shards, axis=0)
    except ValueError as array_split_error:
      raise ValueError(
          f'Unable to put to devices shape {x.shape} with '
          f'local device count {local_device_count}'
      ) from array_split_error

    # Construct mapping of device to index in the split local array.
    device_to_split_array_idx = {}
    i = 0
    for idx in host_to_idxs[jax.process_index()]:
      assert idx in idx_to_devices
      for d in idx_to_devices[idx]:
        device_to_split_array_idx[d] = i % len(local_array_shards)
      i += 1

    device_buffers = []
    for d in local_devices:
      assert d in device_to_split_array_idx
      i = device_to_split_array_idx[d]
      device_buffers.append(jax.device_put(local_array_shards[i], d))

    return device_buffers

  device_buffers = jax.tree_map(_put_to_devices, host_arrays, global_shapes)

  def _jax_array(dbs, global_shape):
    return jax.make_array_from_single_device_arrays(
        global_shape, jax.sharding.NamedSharding(global_mesh, axes), dbs
    )

  return jax.tree_map(
      _jax_array,
      device_buffers,
      global_shapes,
      is_leaf=lambda x: isinstance(x, (list, tuple)),
  )


class ShardedDatasetIterator(clu.data.dataset_iterator.DatasetIterator):
  """A wrapper iterator that returns sharded arrays."""

  def __init__(
      self,
      iterator: clu.data.dataset_iterator.DatasetIterator,
      partitioner: Any,
      global_shapes: PyTree,
  ):
    self._iterator = iterator
    self._global_shapes = pop_metadata(global_shapes)[0]
    self._partitioner = partitioner

  def __next__(self):
      # metadata is only used by the evaluator, so we don't shared it
      data, metadata = pop_metadata(next(self._iterator))
      data = _create_sharded_array(self._partitioner, self._global_shapes, data)
      data.update(metadata)
      return data

  def reset(self):
    return self._iterator.reset()

  @property
  def element_spec(self):
    return self._iterator.element_spec

  def save(self, filename):
    return self._iterator.save(filename)

  def restore(self, filename):
    return self._iterator.restore(filename)

  @property
  def iterator(self):
    if isinstance(self._iterator, clu.data.dataset_iterator.TfDatasetIterator):
      return self._iterator.iterator
    return self._iterator


def _copy_to_host_async(x):
    if hasattr(x, 'addressable_data'):
        # Array is fully replicated.
        x.addressable_data(0).copy_to_host_async()
        return x.addressable_data(0)
    else:
        x.copy_to_host_async()
        return x


def prepare_train_iter(
    train_iter: Union[
        tf.data.Dataset, clu.data.dataset_iterator.DatasetIterator
    ],
    *,
    partitioner,
    data_layout,
) -> clu.data.dataset_iterator.PeekableDatasetIterator:
  """Prepares the training input iterator."""
  # FIXME import this is causing installation for me atm
  # if isinstance(train_iter, PyGrainDatasetIteratorWrapper):
  #   return train_iter
  if isinstance(train_iter, tf.data.Dataset):
    train_iter = clu.data.dataset_iterator.TfDatasetIterator(
        train_iter, checkpoint=True
    )
  elif not isinstance(train_iter, clu.data.dataset_iterator.DatasetIterator):
    raise ValueError(
        f'get_dataset_fn returned unsupported type {type(train_iter)}.'
    )
  input_shapes = jax.tree_map(
      lambda x: (data_layout.batch_size, *x.shape[1:]), train_iter.element_spec
  )
  train_iter = ShardedDatasetIterator(train_iter, partitioner, input_shapes)
  return clu.data.dataset_iterator.PeekableDatasetIterator(train_iter)

@dataclasses.dataclass
class DataLayout:
  """Represents data layout for the partitioned model."""
  batch_size: int
  shard_id: int
  num_shards: int
  is_first_host_in_replica_set: bool

class DataPartitioner(metaclass=abc.ABCMeta):
  def __init__(self, mesh):
    self.mesh = mesh
    self._data_axis = 'dp'
    
  @property
  def data_mesh_size(self) -> int:
    """Data mesh size.

    Data mesh size is defined as the number of global devices involved to
    carry out data parallel. Let's say we have a global mesh: ('replica': 2,
    'data': 4, 'model': 2), and axes 'replica' and 'data' are responsible for
    the data parallel, that means we have 2*4 = 8 devices involved - i.e., data
    mesh size is 8.

    Returns:
      the id of the shard for the axes being replicated among the devices used
      to shard the sharded_mesh_axes.
    """
    data_submesh_sizes = (
        [self.mesh.shape[self._data_axis]]
        if isinstance(self._data_axis, str)
        else [self.mesh.shape[axis] for axis in self._data_axis]
    )
    data_mesh_size = functools.reduce(lambda x, y: x * y, data_submesh_sizes)
    return data_mesh_size

  @property
  def data_partition_spec(self) -> PS:
    return PS(self._data_axis)

  @property
  def data_shards(self) -> int:
    """Number of data shards.
    
    Let's say we are dealing with 2 slices of df4x2 TPUs. In data pipeline
    we need prepare / send one data shard to each local host. This means, we
    need 4 shards since we have 4 local hosts. How to infer the number of hosts
    from mesh information? In this case, we have a global mesh: ('replica': 2,
    'data': 8, 'model': 2). Each local host (i.e., df2x2) has this local mesh:
    ('replica': 1, 'data': 4, 'model': 2). By dividing global mesh with local
    mesh, we can get the count of hosts.

    Returns:
      Number of data shards. Each shard will be sent to one local host.
    """
    data_chunks = (
        [self._local_chunker.num_chunks[self._data_axis]]
        if isinstance(self._data_axis, str)
        else [self._local_chunker.num_chunks[axis] for axis in self._data_axis]
    )
    data_shards = functools.reduce(lambda x, y: x * y, data_chunks)
    return data_shards

  @property
  def data_shard_id(self) -> int:
    """Data shard id for the current host.

    Returns:
      Index of data shard that will be sent to the current local host.
    """
    return self._local_chunker.get_shard_id(self._data_axis)

  @property
  def _local_chunker(self):
    return LocalChunker(self.mesh)

  def get_data_layout(
      self, batch_size: Optional[int] = None, host_index: Optional[int] = None
  ) -> DataLayout:
    """Returns filled `DataLayout` based on the partitioned model layout.

    Args:
      batch_size: if set, indicates the requested batch size. The exception will
        be raised if this batch size is not compatible with the layout. If not
        set, the batch size is inferred from the layout.
      host_index: indicates the host index to use for the calculations, if not
        set - use JAX-provided one. Should be in [0, num_hosts) interval and the
        order should match the order of corresponding CPU devices in
        `jax.devices()`.

    Returns:
      Filled `DataLayout` structure.
    """
    if host_index is not None:
      raise NotImplementedError('Explicit host_index is not yet implemented.')
    if self._data_axis is None:
      return DataLayout(
          batch_size=batch_size,
          shard_id=0,
          num_shards=1,
          is_first_host_in_replica_set=(jax.process_index() == 0))

    batch_size = batch_size or self.data_mesh_size
    if batch_size % self.data_mesh_size:
      raise ValueError(
          f'Batch size ({batch_size}) must be divisible by corresponding '
          f'data mesh size ({self.data_mesh_size}).'
      )

    if batch_size % self.data_shards:
      raise ValueError(
          f'Batch size ({batch_size}) must be divisible by number of '
          f'data shards ({self.data_shards}).'
      )
    replica_id = self._local_chunker.get_replica_id(self._data_axis)
    
    return DataLayout(
        batch_size=int(batch_size),
        shard_id=int(self.data_shard_id),
        num_shards=int(self.data_shards),
        is_first_host_in_replica_set=(replica_id == 0),
    )
    

# Data chunking helper.
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class LocalChunkInfo:
  # The logical slice of an array located on this host's local devices.
  slice: Tuple[slice, ...]
  # A unique index for this host/local chunk among chunks with the same slice.
  replica_id: int

class LocalChunker:
  """Utility class to aid chunking of sharded arrays in multihost settings."""

  def __init__(self, global_mesh: Mesh):
    self.global_mesh = global_mesh
    local_mesh = global_mesh.local_mesh
    first_local_device = local_mesh.devices.reshape(-1)[0]
    host_location = collections.OrderedDict(
        zip(
            global_mesh.shape.keys(),
            list(zip(*np.nonzero(
                global_mesh.devices == first_local_device)))[0]))
    self.num_chunks = collections.OrderedDict()
    self.chunk_ids = collections.OrderedDict()
    self.mesh_axes = list(global_mesh.shape.keys())
    for mesh_axis in self.mesh_axes:
      num_devices_per_chunk = local_mesh.shape[mesh_axis]
      self.num_chunks[mesh_axis] = (
          global_mesh.shape[mesh_axis] // num_devices_per_chunk)
      self.chunk_ids[mesh_axis] = (
          host_location[mesh_axis] // num_devices_per_chunk)

  def get_local_chunk_info(
      self, global_shape: Tuple[int, ...],
      mesh_axes: Sequence[Optional[str]]) -> LocalChunkInfo:
    """Get the local chunk info for a given array shape and sharded axes.

    Args:
      global_shape: the global, unsharded shape of the array to chunk.
      mesh_axes: a sequence of names (or None) of equal rank to `global_shape`
        that specifies which mesh dimensions the array is sharded along.

    Returns:
      LocalChunkInfo containing the logical slices of the array found on this
      host's local devices, as well as the replica index for this chunk among
      chunks with the same slice. The latter is used to determine which
      host should write this chunk during checkpointing.
    """
    local_slice = [slice(None) for dim in global_shape]
    sharded_mesh_axes = set()

    for i, (mesh_axis, size) in enumerate(zip(mesh_axes, global_shape)):
      if not mesh_axis:
        continue
      sharded_mesh_axes.add(mesh_axis)
      if not isinstance(mesh_axis, str):
        raise NotImplementedError('TODO(jekbradbury)')
      chunk_id = self.chunk_ids[mesh_axis]
      chunk_size = size // self.num_chunks[mesh_axis]
      local_slice[i] = slice(chunk_id * chunk_size, (chunk_id + 1) * chunk_size)

    replica_id = self.get_replica_id(sharded_mesh_axes)

    return LocalChunkInfo(tuple(local_slice), replica_id)

  def get_shard_id(self, sharded_mesh_axes: str | Set[Optional[str]]) -> int:
    """Given mesh axes used for sharding, computes current host's shard id.

    To give an example, let's say there are two axes globally: replica, data,
    and model, the mesh axes for sharding is ('replica', 'data'), which means we
    are going to partition an array along 'replica' and 'data' axes.
    The shard_id is to show the index of the current local host along the
    sharding axes (in this example, it's 'replica' and 'data' axes).

    More concretely, let's say we have 4 local hosts, and we use 'replica' and
    'data' axes for data parallel (2 hosts along the replica axis, and 2 host
    along the data axis). The host located in ('replica': 0, 'data': 0), we
    should assign data shard-0 to it. For host ('replica': 0, 'data': 1), we
    assign shard-1. For host ('replica': 1, 'data': 0), we assign shard-2.
    For host ('replica': 1, 'data': 1), we assign shard-3.

    Note: the host location along 'replica' and 'data' axes, e.g.,
    ('replica': 0, 'data': 0) is named chunk_id and stored in
    self._local_chunker.chunk_ids[axis].

    Args:
      sharded_mesh_axes: the mesh axes for sharding.

    Returns:
      the index of the current local host along the sharding axes.
    """
    if isinstance(sharded_mesh_axes, str):
      sharded_mesh_axes = (sharded_mesh_axes,)

    shard_id = 0
    for mesh_axis in sharded_mesh_axes:
      chunk_id = self.chunk_ids[mesh_axis]
      shard_id = shard_id * self.num_chunks[mesh_axis] + chunk_id

    return shard_id

  def get_replica_id(self, sharded_mesh_axes: str | Set[Optional[str]]) -> int:
    """Given mesh axes used for sharding, computes current host's replica id.

    To give an example, let's say there are two axes globally: data, and model,
    the mesh axes for sharding is ('data', ), which means we are going to
    partition an array along 'data' axis and replicate it along 'model' axis.
    The replica_id is to show the index of the current local host along the
    'model' axis.

    Args:
      sharded_mesh_axes: the mesh axes for sharding.

    Returns:
      the index of the current local host along the non-sharding axes (i.e.,
      replicating axes).
    """
    if isinstance(sharded_mesh_axes, str):
      sharded_mesh_axes = (sharded_mesh_axes,)

    replicated_mesh_axes = [
        mesh_axis for mesh_axis in self.mesh_axes
        if mesh_axis not in sharded_mesh_axes
    ]
    replica_id = 0
    for mesh_axis in replicated_mesh_axes:
      chunk_id = self.chunk_ids[mesh_axis]
      replica_id = replica_id * self.num_chunks[mesh_axis] + chunk_id

    return replica_id

class JaxRNG(object):
    """ A convenient stateful Jax RNG wrapper. Can be used to wrap RNG inside
        pure function.
    """

    @classmethod
    def from_seed(cls, seed):
        return cls(jax.random.PRNGKey(seed))

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, keys=None):
        if keys is None:
            self.rng, split_rng = jax.random.split(self.rng)
            return split_rng
        elif isinstance(keys, int):
            split_rngs = jax.random.split(self.rng, num=keys + 1)
            self.rng = split_rngs[0]
            return tuple(split_rngs[1:])
        else:
            split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
            self.rng = split_rngs[0]
            return {key: val for key, val in zip(keys, split_rngs[1:])}


class JaxDistributedConfig(object):
    """ Utility class for initializing JAX distributed. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.initialize_jax_distributed = False
        config.coordinator_address = placeholder(str)
        config.num_processes = placeholder(int)
        config.process_id = placeholder(int)
        config.local_device_ids = placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def initialize(cls, config):
        config = cls.get_default_config(config)
        if config.initialize_jax_distributed:
            if config.local_device_ids is not None:
                local_device_ids = [int(x) for x in config.local_device_ids.split(',')]
            else:
                local_device_ids = None

            jax.distributed.initialize(
                coordinator_address=config.coordinator_address,
                num_processes=config.num_processes,
                process_id=config.process_id,
                local_device_ids=local_device_ids,
            )


class FlaxTemperatureLogitsWarper(FlaxLogitsWarper):
    """ JIT traceable version of FlaxLogitsWarper that performs temperature scaling."""
    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, input_ids, scores, cur_len):
        return scores / jnp.clip(self.temperature, a_min=1e-8)


def make_shard_and_gather_fns(partition_specs, dtype_specs=None):
    """ Create pytree of sharding and gathering functions from pytree of
        partition specs.
    """
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)

    def make_to_dtype_fn(dtype_spec):
        def to_dtype(tensor):
            if dtype_specs in float_dtypes and getattr(tensor, 'dtype', None) in float_dtypes:
                # Convert all float tensors to the same dtype
                return tensor.astype(dtype_specs)
            elif hasattr(dtype_spec, 'dtype') and hasattr(tensor, 'dtype'):
                return tensor.astype(dtype_spec.dtype)
            return tensor
        return to_dtype

    def make_shard_fn(partition_spec, dtype_spec=None):
        jax_shard_function = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=None,
            out_shardings=partition_spec
        )
        def shard_fn(tensor):
            return jax_shard_function(tensor).block_until_ready()
        return shard_fn

    def make_gather_fn(partition_spec, dtype_spec=None):
        jax_gather_fn = pjit(
            make_to_dtype_fn(dtype_spec),
            in_shardings=partition_spec,
            out_shardings=None
        )
        def gather_fn(tensor):
            return jax.device_get(jax_gather_fn(tensor))
        return gather_fn

    if dtype_specs is None or dtype_specs in float_dtypes:
        shard_fns = jax.tree_util.tree_map(make_shard_fn, partition_specs)
        gather_fns = jax.tree_util.tree_map(make_gather_fn, partition_specs)
    else:
        shard_fns = jax.tree_util.tree_map(
            make_shard_fn, partition_specs, dtype_specs
        )
        gather_fns = jax.tree_util.tree_map(
            make_gather_fn, partition_specs, dtype_specs
        )
    return shard_fns, gather_fns


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def get_jax_mesh(axis_dims, names):
    if axis_dims.startswith('!'):
        # Allow splitting a physical mesh axis if needed
        mesh_axis_splitting = True
        axis_dims = axis_dims[1:]
    else:
        mesh_axis_splitting = False

    if ':' in axis_dims:
        dims = []
        dim_names = []
        for axis in axis_dims.split(','):
            name, dim = axis.split(':')
            assert name in names
            dims.append(int(dim))
            dim_names.append(name)
        assert(set(dim_names) == set(names))
    else:
        dims = [int(x) for x in axis_dims.split(',')]
        dim_names = names
    assert len(dims) == len(names)
    mesh_shape = np.arange(jax.device_count()).reshape(dims).shape
    if mesh_axis_splitting:
        physical_mesh = np.array(jax.devices()).reshape(mesh_shape)
    else:
        physical_mesh = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(physical_mesh, dim_names)


def names_in_current_mesh(*names):
    """ Check if current mesh axes contain these names. """
    mesh_axis_names = pxla.thread_resources.env.physical_mesh.axis_names
    return set(names) <= set(mesh_axis_names)


def get_names_from_parition_spec(partition_specs):
    """ Return axis names from partition specs. """
    names = set()
    if isinstance(partition_specs, dict):
        partition_specs = partition_specs.values()
    for item in partition_specs:
        if item is None:
            continue
        elif isinstance(item, str):
            names.add(item)
        else:
            names.update(get_names_from_parition_spec(item))

    return list(names)


def with_sharding_constraint(x, partition_specs):
    """ A smarter version of with_sharding_constraint that only applies the
        constraint if the current mesh contains the axes in the partition specs.
    """
    axis_names = get_names_from_parition_spec(partition_specs)
    if names_in_current_mesh(*axis_names):
        x = _with_sharding_constraint(x, partition_specs)
    return x


def wrap_function_with_rng(rng):
    """ To be used as decorator, automatically bookkeep a RNG for the wrapped function. """
    def wrap_function(function):
        def wrapped(*args, **kwargs):
            nonlocal rng
            rng, split_rng = jax.random.split(rng)
            return function(split_rng, *args, **kwargs)
        return wrapped
    return wrap_function


def init_rng(seed):
    global jax_utils_rng
    jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
    global jax_utils_rng
    return jax_utils_rng(*args, **kwargs)


def get_metrics(metrics, n_steps):
    fetched_metrics = jax.tree_util.tree_map(jax.device_get, metrics)
    final_metrics = metrics_lib.set_step_metrics_num_steps(fetched_metrics, n_steps)

    def _ensure_not_on_device(x):
        assert not isinstance(x, jax.Array)

    jax.tree_util.tree_map(_ensure_not_on_device, final_metrics)
    final_metrics = jax.tree_util.tree_map(get_local_data, final_metrics)
    return {k: float(v.compute_value().value) for k, v in final_metrics.items()}


def mse_loss(val, target, valid=None):
    if valid is None:
        valid = jnp.ones((*target.shape[:2], 1))
    valid = valid.astype(jnp.float32)
    loss = jnp.mean(
        jnp.where(
            valid > 0.0,
            jnp.square(val - target),
            0.0
        )
    )
    return loss


def cross_entropy_loss_and_accuracy(logits, tokens, valid=None):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    
    valid = valid.astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32) # for numerical stability
    token_log_prob = jnp.squeeze(
        jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1),
            jnp.expand_dims(tokens, -1),
            axis=-1,
        ),
        -1,
    )
    token_log_prob = jnp.where(valid > 0.0, token_log_prob, jnp.array(0.0))
    loss = -jnp.mean(jnp.sum(token_log_prob, axis=-1) / valid_text_length)
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, accuracy

def cross_entropy_loss_zloss_and_accuracy(logits, tokens, valid=None, z_loss_alpha=0.0, label_smoothing=0.0):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    
    valid = valid.astype(jnp.float32) * (valid != -1).astype(jnp.float32)
    valid_text_length = jnp.maximum(jnp.sum(valid, axis=-1), 1e-10)
    logits = logits.astype(jnp.float32) # for numerical stability
  
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))

    soft_targets = common_utils.onehot(tokens, vocab_size, on_value=confidence, off_value=low_confidence)
    padded_targets = jnp.zeros(soft_targets.shape[:2] + (logits.shape[-1] - vocab_size,))
    soft_targets = jnp.concatenate([soft_targets, padded_targets], axis=-1)
    
    loss, z_loss = cross_entropy_with_logits(logits, soft_targets, z_loss_alpha)
    loss = loss - normalizing_constant

    loss = loss * valid
    z_loss = z_loss * valid
    valid_sum = jnp.sum(valid)
    
    loss = jnp.sum(loss) / jnp.maximum(valid_sum, 1)
    z_loss = jnp.sum(z_loss) / jnp.maximum(valid_sum, 1)
    
    correct = jnp.where(
        valid > 0.0,
        jnp.argmax(logits, axis=-1) == tokens,
        jnp.array(False)
    )
    accuracy = jnp.mean(jnp.sum(correct, axis=-1) / valid_text_length)
    return loss, z_loss, accuracy


def cross_entropy_loss_zloss(logits, tokens, valid=None, z_loss_alpha=0.0, label_smoothing=0.0):
    if valid is None:
        valid = jnp.ones(tokens.shape[:2])
    
    logits = logits.astype(jnp.float32) # for numerical stability
  
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))

    soft_targets = common_utils.onehot(tokens, vocab_size, on_value=confidence, off_value=low_confidence)
    padded_targets = jnp.zeros(soft_targets.shape[:2] + (logits.shape[-1] - vocab_size,))
    soft_targets = jnp.concatenate([soft_targets, padded_targets], axis=-1)
    
    loss, z_loss = cross_entropy_with_logits(logits, soft_targets, z_loss_alpha)
    loss = loss - normalizing_constant

    loss = loss * valid
    z_loss = z_loss * valid
    valid_sum = jnp.sum(valid)
    
    loss = jnp.sum(loss) / jnp.maximum(valid_sum, 1)
    z_loss = jnp.sum(z_loss) / jnp.maximum(valid_sum, 1)
    
    return loss, z_loss
  

@jax.custom_vjp
def cross_entropy_with_logits(logits: jnp.ndarray, targets: jnp.ndarray,
                              z_loss: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cross entropy loss with stable custom gradient.

  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)

  If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
  will be added to the cross entropy loss (z = softmax normalization constant).
  The two uses of z_loss are:
  1. To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  2. To encourage the logits to be normalized log-probabilities.

  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical one-hot targets [batch, length, num_classes] float
      array.
    z_loss: coefficient for auxilliary z-loss loss term.

  Returns:
    tuple with the total loss and the z_loss, both
    float arrays with shape [batch, length].
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxilliary z-loss term.
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    z_loss: float = 0.0
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray],
           Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                 jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
  """Forward-mode of `cross_entropy_with_logits`."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxilliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (logits, targets, z_loss, exp_shifted, sum_exp,
                                log_softmax, log_z)


def _cross_entropy_with_logits_bwd(
    res: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
               jnp.ndarray, jnp.ndarray], g: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Backward-mode of `cross_entropy_with_logits`."""
  g = g[0]  # Ignore z_loss component as that is only used for logging.
  logits, targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
  # z-loss term adds the (2 * z_loss * log_z) factor.
  deriv = (
      jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp -
      targets)
  g_logits = jnp.expand_dims(g, axis=-1) * deriv
  g_targets = -jnp.expand_dims(g, axis=-1) * log_softmax
  return (jnp.asarray(g_logits,
                      logits.dtype), jnp.asarray(g_targets, targets.dtype),
          jnp.array(0.0))  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd,
                                 _cross_entropy_with_logits_bwd)



def global_norm(tree):
    """ Return the global L2 norm of a pytree. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))


def average_metrics(metrics):
    return jax.tree_map(
        lambda *args: jnp.mean(jnp.stack(args)),
        *metrics
    )


def get_float_dtype_by_name(dtype):
    return {
        'bf16': jnp.bfloat16,
        'bfloat16': jnp.bfloat16,
        'fp16': jnp.float16,
        'float16': jnp.float16,
        'fp32': jnp.float32,
        'float32': jnp.float32,
        'fp64': jnp.float64,
        'float64': jnp.float64,
    }[dtype]


def float_tensor_to_dtype(tensor, dtype):
    if dtype is None or dtype == '':
        return tensor
    if isinstance(dtype, str):
        dtype = get_float_dtype_by_name(dtype)
    float_dtypes = (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64)
    if getattr(tensor, 'dtype', None) in float_dtypes:
        tensor = tensor.astype(dtype)
    return tensor


def float_to_dtype(tree, dtype):
    return jax.tree_util.tree_map(
        partial(float_tensor_to_dtype, dtype=dtype), tree
    )


def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]


def tree_path_to_string(path, sep=None):
    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def flatten_tree(xs, is_leaf=None, sep=None):
    flattened, _ = jax.tree_util.tree_flatten_with_path(xs, is_leaf=is_leaf)
    output = {}
    for key, val in flattened:
        output[tree_path_to_string(key, sep=sep)] = val
    return output


def named_tree_map(f, tree, *rest, is_leaf=None, sep=None):
    """ An extended version of jax.tree_util.tree_map, where the mapped function
        f takes both the name (path) and the tree leaf as input.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, x, *r: f(tree_path_to_string(path, sep=sep), x, *r),
        tree, *rest,
        is_leaf=is_leaf
    )


def match_partition_rules(rules, params):
    """ Returns a pytree of PartitionSpec according to rules. Supports handling
        Flax TrainState and Optax optimizer state.
    """
    def get_partition_spec(name, leaf):
        if len(leaf.shape) == 0 or np.prod(leaf.shape) == 1:
            """ Don't partition scalar values. """
            return PS()
        for rule, ps in rules:
            if re.search(rule, name) is not None:
                return ps
        raise ValueError(f'Partition rule not found for param: {name}')
    
    return named_tree_map(get_partition_spec, params, sep='/')


def get_weight_decay_mask(exclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """
    def decay(name, _):
        for rule in exclusions:
            if re.search(rule, name) is not None:
                return False
        return True

    def weight_decay_mask(params):
        return named_tree_map(decay, params, sep='/')

    return weight_decay_mask

def get_trainable_params_mask(inclusions):
    """ Return a weight decay mask function that computes the pytree masks
        according to the given exclusion rules.
    """
    def trainable_params(name, _):
        for rule in inclusions:
            if re.search(rule, name) is not None:
                return 'trainable'
        return 'frozen'

    def trainable_params_mask(params):
        return named_tree_map(trainable_params, params, sep='/')

    return trainable_params_mask


def tree_apply(fns, tree):
    """ Apply a pytree of functions to the pytree. """
    return jax.tree_util.tree_map(lambda fn, x: fn(x), fns, tree)


def create_learning_rate_scheduler(
    factors: str = 'constant * linear_warmup * rsqrt_decay',
    base_learning_rate: float = 0.5,
    warmup_steps: int = 1000,
    decay_factor: float = 0.5,
    steps_per_decay: int = 20000,
    steps_per_cycle: int = 100000,
    step_offset: int = 0,
    min_learning_rate: float = 1e-8):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * linear_decay: linear decay from warmup_steps with decay_factor slope. Note
      this option implies 'constant * linear_warmup', and should not be used in
      in conjunction with `constant` or `linear_warmup` factors.
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by '*' that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.
    step_offset: int, an offset that the step parameters to this function are
      relative to.
    min_learning_rate: float, minimum learning rate to output. Useful for cases
      when a decay function is (mis)configured to decay to non-positive values.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step: jnp.ndarray) -> jnp.ndarray:
    """Step to learning rate function."""
    step = jnp.maximum(0, step - step_offset)
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'linear_decay':
        ret *= base_learning_rate * jnp.minimum(
            step / warmup_steps, 1.0 + decay_factor * (warmup_steps - step))
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    ret = jnp.maximum(ret, min_learning_rate)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn
