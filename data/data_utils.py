import abc
import dataclasses
import functools
from typing import Mapping, Optional, Sequence, List
from absl import logging
import clu
import gin

import seqio
from seqio import utils
from seqio.feature_converters import _check_exact_match, _check_lengths

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.image_ops_impl import _ImageDimensions, _CheckAtLeast3DImage, _assert, _is_tensor

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

import data.seqio_tokenizer as vocab
from config import *

DEFAULT_EXTRA_IDS = 0
OutputFeaturesType = Mapping[str, utils.Feature]


@gin.configurable
def get_default_vocabulary(tokenizer_type='llama', has_extra_token=False):
  if tokenizer_type == 'llama':
    return vocab.SentencePieceVocabulary(
      "gs://unified-io-2-us-east/tokenizer/llama_tokenizer.model",
      extra_ids=DEFAULT_EXTRA_IDS,
      reverse_extra_ids=True,
      hack_to_t5_start_tokens=False,
      extra_tokens=EXTRA_TOKENS if has_extra_token else None,
    )
  elif tokenizer_type == 'mistral':
    return vocab.SentencePieceVocabulary(
      "gs://unified-io-2-us-east/tokenizer/mistral_tokenizer.model",
      extra_ids=DEFAULT_EXTRA_IDS,
      reverse_extra_ids=True,
      hack_to_t5_start_tokens=False,
      extra_tokens=EXTRA_TOKENS if has_extra_token else None,
    )
  elif tokenizer_type == 'gemma':
      return vocab.SentencePieceVocabulary(
      "gs://unified-io-2-us-east/tokenizer/gemma_tokenizer.model",
      extra_ids=DEFAULT_EXTRA_IDS,
      reverse_extra_ids=True,
      hack_to_t5_start_tokens=False,
      extra_tokens=EXTRA_TOKENS if has_extra_token else None,
    )
  else:
    raise ValueError('no tokenizer matched.')


def get_special_token_ids(token=None, cache={}):
    if not cache:
        voc = get_default_vocabulary()
        # Not sure why ATM, but the LLaMa tokenizer will add an extra space token
        # if this string starts with a space, while the gemma one needs the leading space
        if "gemma_tokenizer" in voc._sentencepiece_model_file:
            ids = voc.encode(" " + " ".join(EXTRA_TOKENS))
        else:
            ids = voc.encode(" ".join(EXTRA_TOKENS))
        
        assert len(ids) == len(EXTRA_TOKENS)
        cache.update({k: i for k, i in zip(EXTRA_TOKENS, ids)})
    if token is not None:
        return cache[token]
    return dict(cache)


def append_eos_bos(
    dataset: tf.data.Dataset,
    output_features: OutputFeaturesType,
) -> tf.data.Dataset:
  """Appends EOS to output feature token sequences with `add_eos` set to True.

  Respects the `add_eos` field of the seqio.Features in `output_features`.

  Args:
    dataset: a tf.data.Dataset of tokenized examples to preprocess.
    output_features: a mapping of output feature names to Feature objects.

  Returns:
    a tf.data.Dataset of tokenized examples with EOS added to specified output
    features.
  """

  def _maybe_add_eos(key: str, value: tf.Tensor) -> tf.Tensor:
    if key not in output_features or not output_features[key].add_eos:
      return value
    else:
      eos_id = output_features[key].vocabulary.eos_id
      bos_id = output_features[key].vocabulary.bos_id
      return _append_to_innermost_axis(_append_to_innermost_axis(value, eos_id), bos_id)

  return dataset.map(
      lambda ex: {k: _maybe_add_eos(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

def _append_to_innermost_axis(
    tensor: tf.Tensor, scalar: tf.Tensor,
) -> tf.Tensor:
  """Appends `scalar` to each slice in the innermost axis of `tensor`.

  >>> _append_to_innermost_axis([1, 2, 3], -1)
  [1, 2, 3, -1]
  >>> _append_to_innermost_axis([[1, 2], [3, 4]], -1)
  [[1, 2, -1], [3, 4, -1]]
  >>> _append_to_innermost_axis(tf.ragged.constant([[1, 2], [3]]), -1)
  [[1, 2, -1], [3, -1]]

  Args:
    tensor: The tensor that should have a value appended.
    scalar: The value to append.

  Returns:
    A copy of `tensor` with `scalar` appended to each slice along
    the innermost axis.
  """
  if isinstance(tensor, tf.RaggedTensor):
    if tensor.shape.rank > 2:
      return tensor.with_values(
          _append_to_innermost_axis(tensor.values, scalar)
      )
    else:
      return tf.concat([tensor, tf.fill([tensor.nrows(), 1], scalar)], axis=1)
  else:
    ndims = tf.rank(tensor)
    paddings = tf.concat(
        [tf.zeros((ndims - 1, 2), dtype=tf.int32), tf.constant([[0, 1]])],
        axis=0,
    )
    return tf.pad(tensor, paddings=paddings, constant_values=scalar)

def _shift_right_by_one(tensor: tf.Tensor, bos_id: int = 0) -> tf.Tensor:
  """Shift the input tensor to the right by one position without wrapping."""

  if not (tensor.dtype.is_integer or tensor.dtype.is_floating):
    raise ValueError(f"Only numeric types are supported. Got: {tensor.dtype}")
  # tf.roll wraps around the axis.
  rolled = tf.roll(tensor, shift=1, axis=0)

  # Zero out the first position by multiplying with [0, 1, 1, ..., 1].
  depth = tf.shape(tensor)[0]
  mask = tf.one_hot(0, depth=depth, on_value=0, off_value=1, dtype=tensor.dtype)

  # Expand dims of mask to broadcast to rolled.
  dim_expansion = [slice(None, None)] + [None] * (len(rolled.shape) - 1)
  mask = mask[dim_expansion]
  return rolled * mask + (1 - mask) * bos_id

def make_autoregressive_inputs(
    targets: tf.Tensor,
    sequence_id: tf.Tensor = None,
    output_dtype: Optional[tf.dtypes.DType] = None,
    bos_id: int = 0,
) -> tf.Tensor:
  """Generate inputs for an autoregressive model, by shifting the targets.

  Modified from mesh_tensorflow.transformer.transformer.autoregressive_inputs.

  For the first element of each sequence, the returned input id is 0.

  For a "packed" dataset, also pass the sequence_id tensor, which aligns
  with the targets tensor and contains different values for different
  concatenated examples.

  Example for a packed dataset:

  ```
        targets = [3, 8, 2, 9, 2, 5, 4, 2, -1, -1]
    sequence_id = [1, 1, 1, 2, 2, 3, 3, 3, 0, 0]
         inputs = [1, 3, 8, 1, 9, 1, 5, 4, -1, -1]
                            |     |        |
                            These positions are set to 0 if sequence_id is not
                            None.
  ```

  Args:
    targets: a tf.int32 tensor with shape [length].
    sequence_id: an optional tensor with the same shape as targets.
    output_dtype: an optional output data type.
    bos_id: bos id.

  Returns:
    a tensor with dtype tf.int32 and the same shape as targets.
  """
  output_dtype = output_dtype or targets.dtype
  if sequence_id is not None and not sequence_id.dtype.is_integer:
    raise ValueError(
        "The sequence_id should be integer-valued tensors for a packed dataset."
    )
  if sequence_id is not None and len(targets.shape) > 1:
    raise ValueError(
        "Only 1-D sequences are supported with packing. Got a "
        f"packed {len(targets.shape)}-D sequence."
    )

  inputs = _shift_right_by_one(targets, bos_id)
  if inputs.dtype != output_dtype:
    inputs = tf.cast(inputs, output_dtype)

  # We should have a 0 at the beginning of each sequence rather than the
  # shifted EOS (e.g. 1) from the previous sequence.
  if sequence_id is not None:
    not_first_in_sequence = tf.equal(
        sequence_id, _shift_right_by_one(sequence_id)
    )
    not_first_in_sequence = tf.cast(not_first_in_sequence, output_dtype)
    first_ids = tf.cast((1 - not_first_in_sequence) * bos_id, output_dtype)
    inputs = inputs * not_first_in_sequence + first_ids
  return inputs

def trim_and_pack_dataset(
    dataset: tf.data.Dataset,
    feature_lengths: Mapping[str, int],
    use_custom_ops: bool = False,
) -> tf.data.Dataset:
  """Creates a 'packed' version of a dataset on-the-fly.

  Modified from the tensor2tensor library.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.

  Each example in the output dataset represents several examples in the
  input dataset.

  For each key in the input dataset that also exists in `feature_lengths`, two
  additional keys are created:
    <key>_segment_ids: an int32 tensor identifying the parts
       representing the original example.
    <key>_positions: an int32 tensor identifying the position within the
       original example.

  Features that are not in `feature_lengths` will be removed.

  Example:
    Two input examples get combined to form an output example.
    The input examples are:
    {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0], "idx": 0}
    {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1], "idx": 1}
    The output example is:
    {
                   "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
       "inputs_segment_ids": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
         "inputs_positions": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                  "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
      "targets_segment_ids": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
        "targets_positions": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
    }

    0 represents padding in both the inputs and the outputs.

    Sequences in the incoming examples are truncated to length in
    `feature_lengths`, and the sequences in the output examples all have this
    fixed (padded) length. Features not in `features_length` (i.e, "idx") are
    removed.

  Args:
    dataset: a tf.data.Dataset
    feature_lengths: map from feature key to final length. Other features will
      be discarded.
    use_custom_ops: a boolean - custom ops are faster but require a custom-built
      binary, which is not currently possible on cloud-tpu.

  Returns:
    a tf.data.Dataset
  """
  element_spec = dataset.element_spec
  # Make sure that the dataset contains all keys in `feature_lengths`.
  # for k in feature_lengths:
  #   if k not in element_spec:
  #     raise ValueError(
  #         f"Feature '{k}' not found in dataset. Available keys are "
  #         f"{list(element_spec.keys())}"
  #     )
  #   if (
  #       not element_spec[k].shape.is_compatible_with(tf.TensorShape([None]))
  #       and not use_custom_ops
  #   ):
  #     raise ValueError(
  #         f"Features to be packed must be one-dimensional. '{k}' is not.' "
  #         "Consider setting use_custom_ops if you have higher-rank features."
  #     )
  additional_keys = set(feature_lengths) - set(element_spec)
  for k in additional_keys: feature_lengths.pop(k)

  # Warn if there are any additional keys that will be removed.
  additional_keys = set(element_spec) - set(feature_lengths)
  if additional_keys:
    logging.warning(
        "Features not in `features_length` will be removed during packing: %s",
        additional_keys,
    )

  ds = dataset.map(
      lambda x: {k: x[k][:l, ...] for k, l in feature_lengths.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(feature_lengths.values())
  padded_shapes = {k: [-1] for k in feature_lengths}
  for k in feature_lengths: padded_shapes[k].extend(dataset.element_spec[k].shape[1:])

  padding_values = {}
  for k in feature_lengths:
    if element_spec[k].dtype == tf.int32:
      padding_values[k] = -1
    else:
      padding_values[k] = -1.0

  ds = ds.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

  if use_custom_ops:
    ds = _pack_with_custom_ops(ds, feature_lengths)
  else:
    ds = _pack_with_tf_ops(ds, feature_lengths)

  # Set the Tensor shapes correctly since they get lost in the process.
  def _set_shape(x):
    for k, v in x.items():
      if 'targets' in k:
        new_shape = [feature_lengths[_strip_packed_feature_key(k)]]
      else:
        new_shape = [feature_lengths[k]]
      new_shape.extend(v.get_shape()[1:])
      v.set_shape(new_shape)
    return x
  
  return ds.map(_set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)

@tf.function
def sum_except_first_axis(tensor):
    # Compute the sum along all axes except the first
    axes_to_sum = tuple(range(1, len(tensor.shape)))
    return tf.reduce_sum(tensor, axis=axes_to_sum)
  
def _pack_with_tf_ops(
    dataset: tf.data.Dataset, feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  See trim_and_pack_dataset()

  Uses tf.while_loop. Slow.

  Args:
    dataset: a dataset containing padded batches of examples.
    feature_lengths: mapping from feature key to packed length.

  Returns:
    a dataset.
  """
  
  # feature_lengths = {'targets': 2048}
  empty_example = {}
  for k in feature_lengths:
    if k == 'targets':
      for suff in ("", "_positions"):
        empty_example[k + suff] = tf.zeros([0], dtype=tf.int32)
        empty_example[k + suff].set_shape([None])
    else:
        empty_example[k] = tf.zeros((0,) + dataset.element_spec[k].shape[2:], dtype=dataset.element_spec[k].dtype)
        empty_example[k].set_shape(dataset.element_spec[k].shape[1:])
  
  keys_etc = empty_example.keys()
  
  def _write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      if 'targets' in k:
        paddings = [[0, feature_lengths[_strip_packed_feature_key(k)] - tf.shape(partial[k])[0],]]
      else:
        paddings = tf.concat([[[0, feature_lengths[k] - tf.shape(partial[k])[0],]], tf.tile(tf.zeros((1,2)),  [tf.rank(partial[k])-1, 1])], axis=0)
        paddings = tf.cast(paddings, tf.int32)
      pad_out = tf.pad(partial[k],paddings, constant_values=-1,)

      new_outputs[k] = outputs[k].write(outputs[k].size(),pad_out,)
        
    return new_partial, new_outputs


  def pack_batch(x: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Internal function to map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.

    Args:
      x: a single example

    Returns:
      a tf.data.Dataset
    """
    keys = list(feature_lengths)
    partial = empty_example.copy()
    first_key, *_ = keys
    dynamic_batch_size = tf.shape(x[first_key])[0]
    outputs = {}
    for k in keys:
      if k == 'targets':
        outputs[k] = tf.TensorArray(
            tf.int32,
            size=0,
            dynamic_size=True,
            element_shape=[feature_lengths[k]],
        )
        outputs[k + "_positions"] = tf.TensorArray(
            tf.int32,
            size=0,
            dynamic_size=True,
            element_shape=[feature_lengths[k]],
        )
      else:
        outputs[k] = tf.TensorArray(
            dataset.element_spec[k].dtype,
            size=0,
            dynamic_size=True,
            element_shape=feature_lengths[k]+empty_example[k].shape[1:],
        )
  
    for i in tf.range(0, dynamic_batch_size):
      tf.autograph.experimental.set_loop_options(
          shape_invariants=[
              (partial, {k: tf.TensorShape([None]) + empty_example[k].shape[1:] for k in keys_etc}),
              (outputs, {k: tf.TensorShape(None) + empty_example[k].shape[1:] for k in keys_etc}),
          ]
      )
      
      can_append = True
      one_example = {}
      for k in keys:
        if k == 'targets':
          val = tf.cast(x[k][i], tf.int32)
          val = val[: tf.reduce_sum(tf.cast(tf.not_equal(val, -1), tf.int32))]
        else:
          val = tf.cast(x[k][i], dataset.element_spec[k].dtype)
          val = val[:tf.reduce_sum(tf.cast(sum_except_first_axis(tf.cast(tf.not_equal(val, -1), tf.int32))>0, tf.int32))]
        
        one_example[k] = val
        
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.shape(partial[k])[0] + tf.shape(one_example[k])[0],
                feature_lengths[k],
            ),
        )

      if not can_append:
        partial, outputs = _write_packed_example(partial, outputs)
        
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][: feature_lengths[k]]
        new_seq_len = tf.size(new_seq)

        # update with the new image index encoding.
        if k == 'image_input_idx':
          new_seq += tf.size(partial['targets'])
          
        new_partial[k] = tf.concat([partial[k], new_seq], 0)        
        if k == 'targets':
          new_partial[k + "_positions"] = tf.concat(
              [partial[k + "_positions"], tf.range(new_seq_len, dtype=tf.int32)],
              0,
          )          
      partial = new_partial
    
    partial, outputs = _write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
        if k == 'targets':
          packed[k + "_segment_ids"] = tf.cumsum(
              tf.cast(tf.equal(packed[k + "_positions"], 0), tf.int32), axis=1
          ) * tf.cast(tf.not_equal(packed[k], -1), tf.int32)

    return packed

  dataset = dataset.map(
      pack_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  return dataset.unbatch()

def _strip_packed_feature_key(key: str) -> str:
  strip_suffix = lambda k, s: k[: -len(s)] if k.endswith(s) else k
  return strip_suffix(strip_suffix(key, "_positions"), "_segment_ids")


def _pack_with_custom_ops(
    dataset: tf.data.Dataset, feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  See trim_and_pack_dataset()

  Relies on custom ops which require a custom compiled binary.
  Faster than _pack_with_tf_ops(), and denser packing.

  Args:
    dataset: a dataset containing padded batches of examples.
    feature_lengths: mapping from feature key to packed length.

  Returns:
    a dataset.
  """
  # TODO(adarob): Move ops into this library and fix int64 issue.
  from tensor2tensor.data_generators.ops import pack_sequences_ops  # pylint: disable=g-import-not-at-top

  keys = list(feature_lengths)
  use_generic_custom_ops = False
  if len(keys) == 1:
    (k1,) = keys
    k2 = k1
  elif len(keys) == 2:
    k1, k2 = keys
  else:
    use_generic_custom_ops = True
    logging.info(
        "`pack_sequences_2` cannot pack more than 2 features. "
        "Using `pack_sequences_k` instead."
    )

  element_spec = dataset.element_spec
  for k in feature_lengths:
    if not element_spec[k].dtype.is_integer:
      use_generic_custom_ops = True
      logging.info(
          (
              "`pack_sequences_2` cannot pack non-integer feature '%s'. "
              "Using `pack_sequences_k` instead."
          ),
          k,
      )
    if not element_spec[k].shape.is_compatible_with(
        tf.TensorShape([None, None])
    ):
      use_generic_custom_ops = True
      logging.info(
          (
              "`pack_sequences_2` cannot pack higher rank feature '%s'. "
              "Using `pack_sequences_k` instead."
          ),
          k,
      )

def trim_and_pad_dataset(
    dataset: tf.data.Dataset, feature_lengths: Mapping[str, int]
) -> tf.data.Dataset:
  """Trim and pad first dimension of features to `feature_lengths`.

  Args:
    dataset: tf.data.Dataset, the dataset to trim/pad examples in.
    feature_lengths: map from feature key to final length. Other features will
      be returned unchanged.

  Returns:
    Trimmed/padded tf.data.Dataset.
  """

  def _trim_and_pad(k: str, t: tf.Tensor) -> tf.Tensor:
    """Trim/pad to the first axis of `t` to be of size `length`."""
    if k not in feature_lengths:
      return t
    if isinstance(t, tf.RaggedTensor):
      t = t.to_tensor()

    constant_values = -1
    length_k = feature_lengths[k]
    if isinstance(length_k, int):
      t = t[:length_k]
      pad_amt = length_k - tf.shape(t)[0]
      padded_t = tf.pad(t, [(0, pad_amt)] + [(0, 0)] * (len(t.shape) - 1), constant_values=constant_values)
      padded_t.set_shape([length_k] + t.shape.as_list()[1:])
      return padded_t

    slices = tuple((slice(0, limit) for limit in length_k))
    t = t[slices]
    pad_amt = tf.pad((length_k - tf.shape(t))[..., None], ((0, 0), (1, 0)), constant_values=constant_values)
    padded_t = tf.pad(t, pad_amt, constant_values=constant_values)
    padded_t.set_shape(length_k)
    return padded_t

  return dataset.map(
      lambda x: {k: _trim_and_pad(k, t) for k, t in x.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

def non_padding_position(
    tensor: tf.Tensor, dtype: tf.dtypes.DType = tf.int32, pad_id: List[int] = [-1]
) -> tf.Tensor:
  """Return a tensor with 1 on non-padding and 0 on padding positions."""
  assert len(pad_id) == 2  
  return tf.cast(tf.logical_and(tf.not_equal(tensor, pad_id[0]), tf.not_equal(tensor, pad_id[1])), dtype=dtype)


class FeatureConverter(seqio.feature_converters.FeatureConverter):
  """Abstract base class for feature converters.

  Subclasses of FeatureConverter are used to convert the tf.data.Dataset
  instance from the Task API to features that are passed to the
  model implementation. Note that Task API has an attribute "output_features",
  which is referred to as "task features" in the context of FeatureConverter.

  Typically the task features contain keys: "inputs" and "targets". The model
  features are constructed based on what is consumed by the model architecture.
  For custom model architectures that require additional model features, one
  needs to subclass FeatureConverter.

  This conversion is fully specified by

    1. defining the mapping of the features in `_convert_features` method and
    2. defining the relationship between sequence lengths of input and output
       features in `get_model_feature_lengths` which is a function of
       task_feature_lengths.

  Therefore, a subclass of FeatureConverter should override `_convert_features`
  and `get_model_feature_lengths` methods.

  The actual feature conversion is done in the `__call__` method, which
  wraps around the `_convert_features` method in order to provide useful checks
  and ensure compatibilities. See `_validate_dataset` and `__call__` methods
  for more details.

  Other notes:

    If pack = True, each feature in the task features should be packable,
    i.e., 1-dimensional.

    Subclasses must override TASK_FEATURES and MODEL_FEATURES. If packing is
    used, they must override PACKING_FEATURE_DTYPES as well. These are the
    packing-specific features such as "*_segment_ids".

    Pass-through features are incompatible with packing and should not be used
    in that case. FeatureConverter only implements the scaffolding, but the real
    pass-through should be implemented in each sub-class inside
    `_convert_features` and `get_model_feature_lengths`.

  Attributes:
    pack: whether to pack the dataset.
    use_custom_packing_ops: whether to use custom ops for packing.
    apply_length_check: if True, it checks whether output feature lengths are
      less than the lengths given by `sequence_length`.
    bos_id: bos id for decoder inputs.
    passthrough_features: a mapping that extends the `TASK_FEATURES` and
      `MODEL_FEATURES` including features that will pass through without any
      processing.
  """

  @dataclasses.dataclass(frozen=True)
  class FeatureSpec:
    """Rank and dtype specifications for features."""

    dtype: tf.dtypes.DType
    rank: int = 1
    sequence_dim: int = 0

  TASK_FEATURES: Mapping[str, "FeatureConverter.FeatureSpec"]
  MODEL_FEATURES: Mapping[str, "FeatureConverter.FeatureSpec"]
  PACKING_FEATURE_DTYPES: Mapping[str, tf.dtypes.DType]

  def __init__(
      self,
      pack: bool = True,
      use_custom_packing_ops: bool = False,
      apply_length_check: bool = True,
      bos_id: int = 1,
      passthrough_features: Optional[
          Mapping[str, "FeatureConverter.FeatureSpec"]] = None,
      passthrough_metadata=True,
      learned_prompt=None
  ):
    assert bos_id == get_default_vocabulary().bos_token_id
    self._pack = pack
    self._use_custom_packing_ops = use_custom_packing_ops
    self._apply_length_check = apply_length_check
    self._bos_id = bos_id
    self._passthrough_metadata = passthrough_metadata
    self._learned_prompt = learned_prompt

    if self.TASK_FEATURES is None:
      raise ValueError("TASK_FEATURES must be defined in the subclass.")

    if self.MODEL_FEATURES is None:
      raise ValueError("MODEL_FEATURES must be defined in the subclass.")

    if self.pack and self.PACKING_FEATURE_DTYPES is None:
      raise ValueError(
          "PACKING_FEATURE_DTYPES must be defined in the subclass if pack=True."
      )

    if passthrough_features is not None:
      if self.pack:
        raise ValueError("Packing is incompatible with pass-through features.")
      self._passthrough_features = passthrough_features
    else:
      self._passthrough_features = {}

  def _validate_dataset(
      self,
      ds: tf.data.Dataset,
      expected_features: Mapping[str, "FeatureConverter.FeatureSpec"],
      expected_lengths: Mapping[str, int],
      strict: bool,
      error_label: str,
  ) -> tf.data.Dataset:
    """Validate properties of the dataset, raising Exceptions if needed.

    This method is used to validate whether the input dataset is compatible
    with the desired specifications. In particular, the following aspects are
    checked.

    Each feature in `expected_features`
      - is also in `ds`
      - is also in expected_lengths
      - is compatible with the expected lengths

    The compatibility of the length is controlled by strict. If true, the length
    of each feature should exactly match the expected length whereas false
    condition allows the length to be less than or equal to the expected length.

    Args:
      ds: a tf.data.Dataset to be validated
      expected_features: expected features
      expected_lengths: a mapping from feature to its length
      strict: whether the lengths should be strictly equal or a length less than
        or equal to expected length is allowed.
      error_label: a label used to indicate the validation stage

    Returns:
      ds: the same dataset as but with the assertion ops attached.
    """
    element_spec = ds.element_spec
    for feat in expected_features:
      if feat not in element_spec:
        raise ValueError(
            "Dataset is missing an expected feature during "
            f"{error_label} validation: '{feat}'"
        )

      if expected_features[feat].dtype != element_spec[feat].dtype:
        actual_dtype = element_spec[feat].dtype.name
        raise ValueError(
            f"Dataset has incorrect type for feature '{feat}' during "
            f"{error_label} validation: Got {actual_dtype}, expected "
            f"{expected_features[feat].dtype.name}"
        )

      if expected_features[feat].rank != len(element_spec[feat].shape):
        actual_rank = len(element_spec[feat].shape)
        raise ValueError(
            f"Dataset has incorrect rank for feature '{feat}' during "
            f"{error_label} validation: "
            f"Got {actual_rank}, expected {expected_features[feat].rank}"
        )

    sequence_axis_mapping = {
        feat: expected_features[feat].sequence_dim for feat in expected_features
    }
    # Remove rank-0 features from expected lengths to bypass the length check.
    expected_lengths = {
        k: v
        for k, v in expected_lengths.items()
        if k in expected_features and expected_features[k].rank != 0
    }
    if self._apply_length_check:
      ds = _check_lengths(
          ds, expected_lengths, sequence_axis_mapping, strict, error_label
      )
    else:
      logging.info(
          "Length validation is skipped since `apply_length_check=False`"
      )
    return ds

  def __call__(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    r"""Convert the features of `ds` into output features.

    This method should not be overridden by subclasses.

    There are two conversion steps and five validation steps.

    Conversion 1: task features are converted to model features in
                  `_convert_features

    Conversion 2: task_feature_lengths are converted to model_feature_lengths in
                  `get_model_feature_lengths`

    Validation 1: verifies that the user input `task_feature_lengths` only
                  contains the required features.

    Validation 2: verifies whether the input dataset has same or more features,
                  same dtype, and length that is less than or equal compared to
                  input_ds.

    Validation 3: partially verifies the behavior of overridden
                  `get_model_feature_lengths`.

    Validation 4: check whether the output dataset has expected features (extra
                  features are allowed), dtype, rank and lengths (exact match).

    Validation 5: check one-to-one match between the output dataset and
                  `expected_dtypes`. Extra features are not allowed.

    The following diagram describes the validation and conversion processes. We
    treat features in the TASK_FEATURES and MODEL_FEATURES specified as class
    variables as the ground-truth. For validations 3, 4 and 5, we define
    `expected_dtypes`.

    There are 5 validation steps. features (<=) means that features of the
    variable on the left is a subset of those of the variable on the right. For
    example, validation 2 guarantees that TASK_FEATURES has features that are a
    subset of the features of input_ds. Validation 4 has length (==), which
    means that it ensures that each feature in MODEL_FEATURES has the same
    length as the corresponding feature in output_ds.

    Overall, these 5 validations ensures that the output_ds has the expected
    features with exact length, dtype and rank. Again, these validations assume
    that TASK_FEATURES and MODEL_FEATURES are correct.


                        Validation 1                     Validation 2
    task_feature_lengths <-----------> TASK_FEATURES <----------------> input_ds
    |                   features (==)                    features (<=)        |
    |                                                    dtype (==)           |
    |                                                    length (<=)          |
    |                                                    rank (==1)           |
    |                                                                         |
    |   Conversion 2                                           Conversion 1   |
    | get_model_feature_lengths                             _convert_features |
    |                                                                         |
    |                                              Validation 5               |
    |                                           <-------------------->        |
    |                                                 features (==)           |
    |                                                                         |
    \/                    Validation 3                    Validation 4        \/
    model_feature_lengths <-------> expected_dtypes <----------------> output_ds
                        features (==)                     features (<=)
                                                          dtype (==)
                                                          length (==)
                                                          rank (==1)

    Args:
      ds: a tf.data.Dataset to be validated
      task_feature_lengths: a mapping from a task feature to its length

    Returns:
      ds: the converted dataset.
    """
    # Validation 1
    task_features_with_passthrough = dict(self.TASK_FEATURES)
    task_features_with_passthrough.update(self._passthrough_features)
    
    # Comment this check
    # _check_exact_match(
    #     expected_features=list(task_features_with_passthrough),
    #     actual_features=list(task_feature_lengths),
    #     expected_feature_source="TASK_FEATURES",
    #     actual_feature_source="task_feature_lengths",
    # )

    # Validation 2
    ds = self._validate_dataset(
        ds,
        expected_features=task_features_with_passthrough,
        expected_lengths=task_feature_lengths,
        # Before pack/pad, check feature (of ds) length <= task feature length
        strict=False,
        error_label="input_validation",
    )

    # Conversion 1: implemented by subclass
    ds = self._convert_features(ds, task_feature_lengths)

    expected_features = dict(self.MODEL_FEATURES)
    expected_features.update(self._passthrough_features)
    if self.pack:
      for k, v in expected_features.items():
        # Packing requires rank 1.
        if v.rank != 1 and not self._use_custom_packing_ops:
          raise ValueError(
              "When packing is enabled, expected ranks must be 1 or "
              f"use_custom_packing_ops must be set. Got expected rank {v.rank} "
              f"for feature {k}."
          )
      for k, v in self.PACKING_FEATURE_DTYPES.items():
        expected_features[k] = FeatureConverter.FeatureSpec(rank=1, dtype=v)

    # Conversion 2: implemented by subclasses
    model_feature_lengths = self.get_model_feature_lengths(task_feature_lengths)

    # Validation 3
    _check_exact_match(
        expected_features=list(expected_features),
        actual_features=list(model_feature_lengths),
        expected_feature_source="model_feature_names",
        actual_feature_source="model_feature_lengths",
    )

    # Validation 4
    ds = self._validate_dataset(
        ds,
        expected_features=expected_features,
        expected_lengths=model_feature_lengths,
        # After pack/pad, check feature (of ds) length == model feature length
        strict=True,
        error_label="output_validation",
    )

    # Validation 5
    # _check_exact_match(
    #     expected_features=list(expected_features),
    #     actual_features=list(ds.element_spec),
    #     expected_feature_source="model_feature_names",
    #     actual_feature_source="output_dataset",
    # )
    return ds

  def _pack_or_pad(
      self, ds: tf.data.Dataset, packed_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Trim/pad to packed_lengths and optionally pack the input dataset."""
    if self.pack:
      ds = trim_and_pack_dataset(
          ds, packed_lengths, self._use_custom_packing_ops
      )
    else:
      ds = trim_and_pad_dataset(ds, packed_lengths)

    return ds

  @abc.abstractmethod
  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Main feature conversion method to be overridden.."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    raise NotImplementedError

  @property
  def pack(self) -> bool:
    return self._pack

  @property
  def bos_id(self) -> int:
    return self._bos_id


def assert_not_truncated(ds, keys, max_val):
    def _check(ex):
        for k in keys:
            tf.assert_less(tf.shape(ex[k])[0], max_val+1,
                           message=f"Field {k} was unexpectedly truncated max_len={max_val}")
        return ex
    return ds.map(_check)


class MultiModalLMFeatureConverter(FeatureConverter):
  """Feature converter for a language model (decoder-only) architecture.

  The input dataset must have "targets" field only.

  One common usecase is to pre-train a decoder-only model with the standard
  language modeling objective (i.e., predict the next token given the previous
  ones) on a unlabeled text corpus which only has "targets". Then the
  pre-trained model can be fine-tuned on a supervised task, e.g., machine
  translation by concatenating "inputs" and "targets". For this use case,
  pre-train with LMFeatureConverter and fine-tune with PrefixLMFeatureConverter.

  Example: a packed dataset.

    ds = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]

    input_lengths = {"targets": 6}

    converted_ds = {
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0],
         "decoder_input_tokens": [0, 3, 9, 0, 4, 0],
         "decoder_loss_weights": [1, 1, 1, 1, 1, 0],
            "decoder_positions": [0, 1, 2, 0, 1, 0],
          "decoder_segment_ids": [1, 1, 1, 2, 2, 0]
    }
  Note that two examples are packed together into one example.
  """

  TASK_FEATURES = {"targets": FeatureConverter.FeatureSpec(dtype=tf.int32)}
  MODEL_FEATURES = {
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  
  PACKING_FEATURE_DTYPES = {
      "decoder_segment_ids": tf.int32,
      "decoder_positions": tf.int32,
  }

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Convert an LM example into an example with model features."""
    # targets_segment_id is present only for a packed dataset.
    if self._learned_prompt:
        logging.info("Adding learned prmopt")
        features["targets"] = tf.concat([
            tf.zeros(self._learned_prompt, dtype=features["targets"].dtype),
            features["targets"][:-self._learned_prompt],
        ], 0)
        features["decoder_loss_weights"] = tf.concat([
            tf.zeros(self._learned_prompt, dtype=features["decoder_loss_weights"].dtype),
            features["decoder_loss_weights"][:-self._learned_prompt],
        ], 0)
        features["image_input_idx"] += self._learned_prompt

    decoder_input_tokens = make_autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segment_ids", None),
        bos_id=self.bos_id,
    )

    special_tokens = tf.constant(list(get_special_token_ids().values()))
    tf.assert_equal(
        True,
        tf.reduce_all(decoder_input_tokens[-1] != special_tokens),
        message="An input ends with an image special token",
    )
    d = {
        "decoder_target_tokens": features["targets"],
        "decoder_input_tokens": decoder_input_tokens,
        "decoder_loss_weights": features["decoder_loss_weights"],
        "images": features["images"],
        # plus one sine we have added BOS to the inputs
        "image_input_idx": features["image_input_idx"] + 1
    }

    if self.pack:
      d["decoder_segment_ids"] = features["targets_segment_ids"]
      d["decoder_positions"] = features["targets_positions"]

    if self._passthrough_metadata:
        for k in features:
            if k.startswith("metadata/"):
                d[k] = features[k]
    return d

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Convert the dataset to be fed to a language model."""
    if "images" in ds.element_spec and "images" in task_feature_lengths:
        # Images should never be truncated
        ds = assert_not_truncated(ds, ["images", "image_input_idx"], task_feature_lengths["images"])
    if "metadata" in ds.element_spec:
        # Inference datasets should not be truncated
        ds = assert_not_truncated(ds, ["targets"], task_feature_lengths["targets"])
    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    decoder_length = task_feature_lengths["targets"]
    model_feature_lengths = {
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length,
    }
    if self.pack:
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["decoder_positions"] = decoder_length

    return model_feature_lengths
  

class LMFeatureConverter(seqio.feature_converters.FeatureConverter):
  """
  Additional modification based on original LMFeatureConverter that can take
  the loss_weights as inputs.
  
  Feature converter for a language model (decoder-only) architecture.

  The input dataset must have "targets" field only.

  One common usecase is to pre-train a decoder-only model with the standard
  language modeling objective (i.e., predict the next token given the previous
  ones) on a unlabeled text corpus which only has "targets". Then the
  pre-trained model can be fine-tuned on a supervised task, e.g., machine
  translation by concatenating "inputs" and "targets". For this use case,
  pre-train with LMFeatureConverter and fine-tune with PrefixLMFeatureConverter.

  Example: a packed dataset.

    ds = [{"targets": [3, 9, 1]}, {"targets": [4, 1]}]

    input_lengths = {"targets": 6}

    converted_ds = {
        "decoder_target_tokens": [3, 9, 1, 4, 1, 0],
         "decoder_input_tokens": [0, 3, 9, 0, 4, 0],
         "decoder_loss_weights": [1, 1, 1, 1, 1, 0],
            "decoder_positions": [0, 1, 2, 0, 1, 0],
          "decoder_segment_ids": [1, 1, 1, 2, 2, 0]
    }
  Note that two examples are packed together into one example.
  """

  TASK_FEATURES = {
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
      # "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {
      "decoder_segment_ids": tf.int32,
      "decoder_positions": tf.int32,
  }

  def _convert_example(
      self, features: Mapping[str, tf.Tensor]
  ) -> Mapping[str, tf.Tensor]:
    """Convert an LM example into an example with model features."""
    # targets_segment_id is present only for a packed dataset.
    decoder_input_tokens = utils.make_autoregressive_inputs(
        features["targets"],
        sequence_id=features.get("targets_segment_ids", None),
        bos_id=self.bos_id,
    )

    if "decoder_loss_weights" in features:
      decoder_loss_weights = features["decoder_loss_weights"]
    else:
      decoder_loss_weights =  tf.cast(features["targets"] > 0, tf.int32)
    
    d = {
        "decoder_target_tokens": features["targets"],
        "decoder_input_tokens": decoder_input_tokens,
        "decoder_loss_weights": decoder_loss_weights,
    }

    if self.pack:
      d["decoder_segment_ids"] = features["targets_segment_ids"]
      d["decoder_positions"] = features["targets_positions"]

    return d

  def _convert_features(
      self, ds: tf.data.Dataset, task_feature_lengths: Mapping[str, int]
  ) -> tf.data.Dataset:
    """Convert the dataset to be fed to a language model."""
    ds = self._pack_or_pad(ds, task_feature_lengths)
    return ds.map(
        self._convert_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  def get_model_feature_lengths(
      self, task_feature_lengths: Mapping[str, int]
  ) -> Mapping[str, int]:
    """Define the length relationship between task and model features."""
    decoder_length = task_feature_lengths["targets"]
    model_feature_lengths = {
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens": decoder_length,
        "decoder_loss_weights": decoder_length,
    }
    if self.pack:
      model_feature_lengths["decoder_segment_ids"] = decoder_length
      model_feature_lengths["decoder_positions"] = decoder_length

    return model_feature_lengths
  
  
def flip_if_vertical(image,is_video=False):
    if is_video:
        return flip_video_if_vertical(image)
    else:
        return flip_image_if_vertical(image)

def flip_image_if_vertical(image):
    """
    https://www.youtube.com/watch?v=f2picMQC-9E
    :param image:
    :return:
    """
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
     # Pad and then add some constants (if it's flipped) to tell the model that we messed with it
    image = tf.cond(
        height >= (4 * width / 3.0),
        lambda: tf.pad(tf.image.rot90(image), [[0,0], [4, 4], [0,0]], mode='CONSTANT', constant_values=0.5),
        lambda: image,
    )
    return image

def flip_video_if_vertical(frames):
    return tf.map_fn(
        fn=flip_image_if_vertical,
        elems=frames)

def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.
    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random.uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


def denormalize_boxes(boxes, image_shape):
    """Converts boxes normalized by [height, width] to pixel coordinates.
    Args:
      boxes: a tensor whose last dimension is 4 representing the coordinates of
        boxes in ymin, xmin, ymax, xmax order.
      image_shape: a list of two integers, a two-element vector or a tensor such
        that all but the last dimensions are `broadcastable` to `boxes`. The last
        dimension is 2, which represents [height, width].
    Returns:
      denormalized_boxes: a tensor whose shape is the same as `boxes` representing
        the denormalized boxes.
    Raises:
      ValueError: If the last dimension of boxes is not 4.
    """
    with tf.name_scope('denormalize_boxes'):
      if isinstance(image_shape, list) or isinstance(image_shape, tuple):
        height, width = image_shape
        height = tf.cast(height, dtype=boxes.dtype)
        width = tf.cast(width, dtype=boxes.dtype)
      else:
        image_shape = tf.cast(image_shape, dtype=boxes.dtype)
        height, width = tf.split(image_shape, 2, axis=-1)

      ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
      ymin = ymin * height
      xmin = xmin * width
      ymax = ymax * height
      xmax = xmax * width

      denormalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
      return denormalized_boxes

def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, value=0):

  return pad_to_bounding_box_internal(
      image,
      offset_height,
      offset_width,
      target_height,
      target_width,
      check_dims=True, 
      value=value)

def pad_to_bounding_box_internal(image, offset_height, offset_width,
                                 target_height, target_width, check_dims, value):

  with ops.name_scope(None, 'pad_to_bounding_box_with_one_internal', [image]):
    image = ops.convert_to_tensor(image, name='image')

    is_batch = True
    image_shape = image.get_shape()
    if image_shape.ndims == 3:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
    elif image_shape.ndims is None:
      is_batch = False
      image = array_ops.expand_dims(image, 0)
      image.set_shape([None] * 4)
    elif image_shape.ndims != 4:
      raise ValueError(
          '\'image\' (shape %s) must have either 3 or 4 dimensions.' %
          image_shape)

    batch, height, width, depth = _ImageDimensions(image, rank=4)

    after_padding_width = target_width - offset_width - width

    after_padding_height = target_height - offset_height - height

    if check_dims:
      assert_ops = _CheckAtLeast3DImage(image, require_static=False)
      assert_ops += _assert(offset_height >= 0, ValueError,
                            'offset_height must be >= 0')
      assert_ops += _assert(offset_width >= 0, ValueError,
                            'offset_width must be >= 0')
      assert_ops += _assert(after_padding_width >= 0, ValueError,
                            'width must be <= target - offset')
      assert_ops += _assert(after_padding_height >= 0, ValueError,
                            'height must be <= target - offset')
      image = control_flow_ops.with_dependencies(assert_ops, image)

    # Do not pad on the depth dimensions.
    paddings = array_ops.reshape(
        tf.stack([
            0, 0, offset_height, after_padding_height, offset_width,
            after_padding_width, 0, 0
        ]), [4, 2])
    padded = array_ops.pad(image, paddings, constant_values=value)

    padded_shape = [
        None if _is_tensor(i) else i
        for i in [batch, target_height, target_width, depth]
    ]
    padded.set_shape(padded_shape)

    if not is_batch:
      padded = array_ops.squeeze(padded, axis=[0])

    return padded

def resize_and_crop_boxes(boxes, image_scale, output_size, offset, paddings):
    """Resizes boxes to output size with scale and offset.
    Args:
      boxes: `Tensor` of shape [N, 4] representing ground truth boxes.
      image_scale: 2D float `Tensor` representing scale factors that apply to
        [height, width] of input image.
      output_size: 2D `Tensor` or `int` representing [height, width] of target
        output image size.
      offset: 2D `Tensor` representing top-left corner [y0, x0] to crop scaled
        boxes.
      paddings: 2D `Tensor` representing top/left paddings.
    Returns:
      boxes: `Tensor` of shape [N, 4] representing the scaled boxes.
    """
    # Adjusts box coordinates based on image_scale, offset and paddings.
    boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
    boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
    boxes += tf.tile(tf.expand_dims(paddings, axis=0), [1, 2])
    # Clips the boxes.
    boxes = clip_boxes(boxes, output_size)
    return boxes

def clip_boxes(boxes, image_shape):
  """Clips boxes to image boundaries.
  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes.
  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  with tf.name_scope('clip_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
      max_length = [height, width, height, width]
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.unstack(image_shape, axis=-1)
      max_length = tf.stack(
          [height, width, height, width], axis=-1)

    clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
    return clipped_boxes
  
  
def get_non_empty_box_indices(boxes):
    """Get indices for non-empty boxes."""
    # Selects indices if box height or width is 0.
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    indices = tf.where(
        tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
    return indices[:, 0]


def resize_and_pad(image, desired_output_size, masks=None, boxes=None, labels=None,
                   random_scale_min=0.1, random_scale_max=2.0, do_random_scale=False,
                   shrink_both_sides=True, boxes1=None, filter_box=True,
                   do_flip_if_vertical=True, desired_target_size=None, random_scale_ratio=0.0,
                   resize_method=tf.image.ResizeMethod.BILINEAR, return_outputs=True, 
                   is_video=False, pad_value=0, normalize=True):
    """
    :param image:
    :param desired_output_size:
    :param boxes:
    :param random_scale_min:
    :param random_scale_max:
    :param do_random_scale:
    :param shrink_both_sides: whether both sides can be shrunk at the same time
    :return:
    """
    if do_flip_if_vertical:
        image = flip_if_vertical(image,is_video)

    desired_height, desired_width = desired_output_size
    desired_height_f = tf.cast(desired_height, dtype=tf.float32)
    desired_width_f = tf.cast(desired_width, dtype=tf.float32)

    if is_video:
      height = tf.cast(tf.shape(image)[1], tf.float32)
      width = tf.cast(tf.shape(image)[2], tf.float32)
    else:
      height = tf.cast(tf.shape(image)[0], tf.float32)
      width = tf.cast(tf.shape(image)[1], tf.float32)

    if boxes is not None:
        # Converts boxes from normalized coordinates to pixel coordinates.
        # Now the coordinates of boxes are w.r.t. the original image.
        boxes = denormalize_boxes(boxes, [height, width])
        
    if boxes1 is not None:
        boxes1 = denormalize_boxes(boxes1, [height, width])

    if do_random_scale:
        random_scale_factor = tf.random.uniform([], random_scale_min, random_scale_max)
        if not shrink_both_sides:
            # Max random is where scale * W > W_desired
            #                     scale * H > H_desired
            rsf_max = tf.maximum(desired_width_f / width, desired_height_f / height)
            random_scale_factor = tf.minimum(rsf_max, random_scale_factor)

        scaled_y = tf.cast(random_scale_factor * desired_height_f, tf.int32)
        scaled_x = tf.cast(random_scale_factor * desired_width_f, tf.int32)

        # Recompute the accurate scale_factor using rounded scaled image size.
        image_scale_y = tf.cast(scaled_y, tf.float32) / height
        image_scale_x = tf.cast(scaled_x, tf.float32) / width
        
        image_scale = tf.cond(tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
            tf.cast(random_scale_ratio, tf.float32)), 
            lambda: tf.maximum(image_scale_x, image_scale_y),
            lambda: tf.minimum(image_scale_x, image_scale_y))
        
        # image_scale = tf.minimum(image_scale_x, image_scale_y)

        # Conceptual captions has some REALLY WIDE images I believe
        # this ensures that we won't scale any side lower than to 64
        image_scale = tf.maximum(image_scale, 64.0 / tf.minimum(height, width))

        # Select non-zero random offset (x, y) if scaled image is larger than
        # self._output_size.
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.cast(scaled_height - desired_height, tf.float32)
        offset_x = tf.cast(scaled_width - desired_width, tf.float32)
        offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform([], 0, 1)
        offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform([], 0, 1)
        offset_y = tf.cast(offset_y, tf.int32)
        offset_x = tf.cast(offset_x, tf.int32)
    else:
        image_scale_y = desired_height_f / height
        image_scale_x = desired_width_f / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.constant(0)
        offset_x = tf.constant(0)

    # Now resize and crop
    if resize_method == 'random' and do_random_scale and (not tf.executing_eagerly()):
        resize_methods = sorted([k for k in tf.image.ResizeMethod.__dict__.keys() if k.isupper()])
        # print("Random resize method:\n{}".format(','.join(resize_methods)))
        image = apply_with_random_selector(
            image,
            lambda x, method_idx: tf.image.resize(x, [scaled_height, scaled_width],
                                                  tf.image.ResizeMethod.__dict__[resize_methods[method_idx]],
                                                  antialias=True),
            num_cases=len(resize_methods))

    elif resize_method != 'random':
        image = tf.image.resize(image, [scaled_height, scaled_width], method=resize_method, antialias=True)
    else:
        print(f"you passed in {resize_method} but doing bilinear resize instead (possibly because eager is on or evaluation is on.)")
        image = tf.image.resize(image, [scaled_height, scaled_width],
                                method=tf.image.ResizeMethod.BILINEAR, antialias=True)

    image = tf.clip_by_value(image, 0.0, 1.0)
    
    if is_video:
        # frames x H x W x C
        image = image[:,offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]
    else:
        # H x W x C
        image = image[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width, :]

    if is_video:
      H = tf.shape(image)[1]
      W = tf.shape(image)[2]
    else:
      H = tf.shape(image)[0]
      W = tf.shape(image)[1]

    top_pad = (desired_height - H) // 2
    left_pad = (desired_width - W) // 2

    if is_video:
        image_mask = pad_to_bounding_box(
            tf.ones_like(image), top_pad, left_pad, desired_height, desired_width)[:,:,:,0]
    else:
        image_mask = pad_to_bounding_box(
            tf.ones_like(image), top_pad, left_pad, desired_height, desired_width)[:,:,0]
    
    image = pad_to_bounding_box(image, top_pad, left_pad, desired_height, desired_width, value=pad_value)

    if isinstance(desired_height, int) and isinstance(desired_width, int):
        if is_video:
            image.set_shape([None, desired_height, desired_width, 3])
        else:
            image.set_shape([desired_height, desired_width, 3])
    else:
        print("Cant set shape bc desired height/width are dynamic")

    if masks is not None and tf.size(masks) != 0:
      masks = tf.image.resize(masks, [scaled_height, scaled_width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

      if len(masks.shape) == 3:
        masks = masks[offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]                    
      else:
        masks = masks[:, offset_y:offset_y + desired_height, offset_x:offset_x + desired_width]                    

      masks = pad_to_bounding_box(masks, top_pad, left_pad, desired_height, desired_width)
      masks = tf.image.resize(masks, desired_target_size,
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      
    indices = None
    if boxes is not None:
      # assert ValueError("the box need to be shift which is not tested yet.")
      boxes = resize_and_crop_boxes(
          boxes, 
          tf.stack([image_scale, image_scale]), 
          [desired_height, desired_width], 
          tf.cast(tf.stack([offset_y, offset_x]), dtype=tf.float32),
          tf.cast(tf.stack([top_pad, left_pad]), dtype=tf.float32))

      if filter_box:
        indices = get_non_empty_box_indices(boxes)
      else:
        indices = tf.range(tf.shape(boxes)[0])
      boxes = tf.gather(boxes, indices)
      
      if labels is not None:
        labels = tf.gather(labels, indices)

      if boxes1 is not None:
        boxes1 = resize_and_crop_boxes(
            boxes1, 
            tf.stack([image_scale, image_scale]), 
            [desired_height, desired_width], 
            tf.cast(tf.stack([offset_y, offset_x]), dtype=tf.float32),
            tf.cast(tf.stack([top_pad, left_pad]), dtype=tf.float32))   

    image_info = tf.stack([
        tf.cast(top_pad, tf.float32),
        tf.cast(left_pad, tf.float32),
        1.0 / image_scale,
        height,
        width,
        tf.cast(offset_y, dtype=tf.float32) / height,
        tf.cast(offset_x, dtype=tf.float32) / width,
        tf.cast(offset_y, dtype=tf.float32),
        tf.cast(offset_x, dtype=tf.float32),
        tf.cast(scaled_height, dtype=tf.float32),
        tf.cast(scaled_width, dtype=tf.float32),
    ])

    if boxes1 is not None:
      outputs = (image_info, masks, boxes, labels, indices, boxes1)
    else:
      outputs = (image_info, masks, boxes, labels, indices)
    
    if normalize:
      if is_video:
        image = normalize_video(image)
      else:
        image = normalize_image(image)
    
    if return_outputs:
      return image, image_mask, outputs
    else:
      return image, image_mask

def _remove_bars_from_frames(frames, black_bar=True, threshold=32, max_perc_to_trim=0.3):
    """
    :param frames: [num_frames, height, width, 3]
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim x% by default of the image at most in each dimension
    :return:
    """
    # Detect black bars####################
    frames_shape = tf.shape(frames)
    h, w = frames_shape[1], frames_shape[2]
    if black_bar:
      has_content = tf.reduce_max(frames, axis=(0, -1)) >= threshold
    else:
      has_content = tf.reduce_min(frames, axis=(0, -1)) <= threshold

    y_frames = tf.cast(tf.reshape(tf.where(tf.reduce_any(has_content, axis=1)), [-1]), tf.int32)
    nhbars = tf.shape(y_frames)[0]
    y_frames = tf.cond(nhbars > 0, lambda: y_frames, lambda: tf.expand_dims(tf.cast(h // 2, tf.int32), axis=0))

    y1 = tf.minimum(y_frames[0], tf.cast(tf.cast(h, tf.float32) * max_perc_to_trim, tf.int32))
    y2 = tf.maximum(y_frames[-1] + 1, tf.cast(tf.cast(h, tf.float32) * (1 - max_perc_to_trim), tf.int32))

    x_frames = tf.cast(tf.reshape(tf.where(tf.reduce_any(has_content, axis=0)), [-1]), tf.int32)
    nvbars = tf.shape(x_frames)[0]
    x_frames = tf.cond(nvbars > 0, lambda: x_frames, lambda: tf.expand_dims(tf.cast(w // 2, tf.int32), axis=0))

    x1 = tf.minimum(x_frames[0], tf.cast(tf.cast(w, tf.float32) * max_perc_to_trim, tf.int32))
    x2 = tf.maximum(x_frames[-1] + 1, tf.cast(tf.cast(w, tf.float32) * (1 - max_perc_to_trim), tf.int32))

    frames = frames[:, y1:y2, x1:x2]
    return frames

def convert_video_dtype(video,dtype):
    """
    Converts tensor to dtype and scales the values. 
    Video equivalent of tf.convert_image_dtype: https://www.tensorflow.org/api_docs/python/tf/image/convert_image_dtype
    """
    return tf.map_fn(
        fn=functools.partial(
            tf.image.convert_image_dtype,
            dtype=dtype),
        elems=video,
        fn_output_signature=dtype)

def _stateless_shuffle(x: tf.Tensor, seed):
  if hasattr(tf.random.experimental, 'stateless_shuffle'):
    return tf.random.experimental.stateless_shuffle(x, seed=seed)
  else:
    vals = tf.random.stateless_uniform(tf.shape(x), seed)
    ixs = tf.argsort(vals)
    return tf.gather(x, ixs)


def sample_patches(mask, n_patches, stateless=False, seeds=None):
  input_sample_valid = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
  input_sample_masked = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask == 0)
  if stateless:
    encoder_pos_ids = tf.concat([
      _stateless_shuffle(input_sample_valid, seeds[0]),
      _stateless_shuffle(input_sample_masked, seeds[1])], axis=0)[:n_patches]
  else:
    encoder_pos_ids = tf.concat([
      tf.random.shuffle(input_sample_valid),
      tf.random.shuffle(input_sample_masked)], axis=0)[:n_patches]
  encoder_pos_ids = tf.reshape(encoder_pos_ids, (n_patches,))
  encoder_pos_ids = tf.cast(encoder_pos_ids, tf.int32)
  return encoder_pos_ids


def normalize_image(image,
                    offset=(0.48145466, 0.4578275, 0.40821073),
                    scale=(0.26862954, 0.26130258, 0.27577711)):
  """Normalizes the image to zero mean and unit variance."""
  offset = tf.constant(offset)
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= tf.cast(offset, image.dtype)

  scale = tf.constant(scale)
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= tf.cast(scale, image.dtype)
  return image


def normalize_video(video,**kwargs):
  return tf.map_fn(
      fn=functools.partial(
        normalize_image,
        **kwargs),
      elems=video)

def unnormalize_image(image,
                    offset=(0.48145466, 0.4578275, 0.40821073),
                    scale=(0.26862954, 0.26130258, 0.27577711)):
  """Normalizes the image to zero mean and unit variance."""
  scale = tf.cast(tf.expand_dims(tf.expand_dims(tf.constant(scale), axis=0), axis=0), image.dtype)
  image *= scale

  offset = tf.cast(tf.expand_dims(tf.expand_dims(tf.constant(offset), axis=0), axis=0), image.dtype)
  image += offset
  return image


def unnormalize_video(video, **kwargs):
  return tf.map_fn(
    fn=functools.partial(
      unnormalize_image,
      **kwargs),
    elems=video)


def flatten_parts(ds: tf.data.Dataset, parts: List[str], add_index = False) -> tf.data.Dataset:
    def _flatten(ex):
        flat_key = {k: ex[k] for k in parts}
        if add_index:
            flat_key['index'] = tf.range(len(ex[parts[0]]))

        flat_ds = tf.data.Dataset.from_tensor_slices(flat_key)

        def _merge(_flat_ex):
            for k, v in ex.items():
                if k not in parts:
                    _flat_ex[k] = v
            return _flat_ex
        return flat_ds.map(_merge)

    return ds.flat_map(_flatten)
