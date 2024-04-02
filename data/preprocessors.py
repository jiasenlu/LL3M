import functools
import math
from functools import reduce
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence, Union

import gin
import seqio
import einops

from absl import logging
from .data_utils import resize_and_pad, get_default_vocabulary, \
    _remove_bars_from_frames, convert_video_dtype, sample_patches, append_eos_bos, \
    get_special_token_ids, flatten_parts, pad_to_bounding_box
import tensorflow as tf

from config import *
from .prompts import * 

FeatureType = Mapping[str, tf.Tensor]
AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_from_dict(data, keys):
    """Iterate nested dictionary"""
    return reduce(dict.get, keys, data)

@seqio.utils.map_over_dataset
def rekey(x, key_map=None):
  """Replace the feature keys according to the mapping in `key_map`.
  For example, if the dataset returns examples of the format:
  {'foo': 'something', 'bar': 'something else'}
  and key_map = {'boo': 'foo', 'spar': 'bar'} then this function will return
  examples with the format
  {'boo': 'something', 'spar': 'something else'}
  If a mapping is to an empty key or None, set the new key to an empty string.
  Args:
      x: an example to process.
      key_map: dictionary mapping new keys to original keys
  Returns:
      A preprocessed example with the format listed above.
  """
  if key_map:
    return {
        new_key: get_from_dict(x, old_key) if old_key else ''
        for new_key, old_key in key_map.items()
    }
  return x

def flan_preprocessor(ds, sequence_length, name, prompt_type = 'llama2'):
  prompt_template = PROPMPT_MANAGER[prompt_type]
  vocab = get_default_vocabulary()

  def to_inputs_and_targets(ex):
    
    prefix = tf.strings.join([prompt_template['B_INST'], prompt_template['SYS_PREFIX'], ex['inputs'], prompt_template['E_INST']], separator=' ')
    encoded_prefix = vocab.encode_tf(prefix)
    prefix_loss_weights = tf.zeros(tf.shape(encoded_prefix), tf.int32)
    encoded_response = vocab.encode_tf(ex['targets'])
    response_loss_weights = tf.ones(tf.shape(encoded_response), tf.int32)
    targets = tf.concat([encoded_prefix, encoded_response], axis=0)
    decoder_loss_weights = tf.concat([prefix_loss_weights, response_loss_weights], axis=0)

    return {
        'targets': targets,
        'decoder_loss_weights': decoder_loss_weights,
    }
    
  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  
  def _filter(ex):
    return tf.math.reduce_sum(ex['decoder_loss_weights']) > 0

  ds = ds.filter(_filter)
  return ds

# -----------------------------------------------------------------------------------------
# NLP tasks
# -----------------------------------------------------------------------------------------

def maybe_cast_seed(seed):
  """Cast seed to int64 if needed."""
  def _maybe_cast_seed(s):
    if s.dtype == tf.int64 or s.dtype == tf.int32:
      return s
    return tf.cast(s, tf.int64)

  # seed is either a Tensor or a pair of Tensors.
  if isinstance(seed, tf.Tensor):
    return _maybe_cast_seed(seed)
  else:
    return _maybe_cast_seed(seed[0]), _maybe_cast_seed(seed[1])

def single_example_select_random_chunk(
    features: FeatureType,
    seed: tf.Tensor,
    output_features: Mapping[str, seqio.Feature],
    max_length: Optional[int] = None,
    feature_key: str = 'targets',
    additional_feature_keys: Optional[Sequence[str]] = None,
    passthrough_feature_keys: Optional[Sequence[str]] = None,
    sequence_length: Optional[Mapping[str, int]] = None,
    uniform_random_start: bool = False,
    min_length: Optional[int] = None) -> FeatureType:
  """Token-preprocessor to extract one span of at most `max_length` tokens.

  If the token sequence is longer than `max_length`, then we return a random
  subsequence.  Otherwise, we return the full sequence.

  This is generally followed by split_tokens.

  Args:
    features: Single example with `feature_key` containing a tokenized sequence.
    seed: Random seed to be used.
    output_features: Mapping of keys to features.
    max_length: Typically specified in gin configs, takes priority over
      sequence_length.
    feature_key: Which feature to use from the dataset.
    additional_feature_keys: Additional features to use. The same chunk will be
      selected from these features as from the one specified in feature_key,
      so they should all have the same length.
    passthrough_feature_keys: Additional keys to pass through unchanged.
    sequence_length: Used if max_length is not specified. Typically passed in
      by the data pipeline. feature_key will be used to select the length.
    uniform_random_start: If True, will select a starting point in
      [-max_length + 1, n_tokens). If False, will select one of a set of chunks
      offset by max_length. Both of these starting points try to ensure each
      token has an equal probability of being included.
    min_length: If specified, lengths of chunks will be selected uniformly at
      random from [min_length, max_length]. Note that chunks can end up shorter
      than min_length if at the beginning or end of the sequence.

  Returns:
    The features of the selected chunk.
  """
  if passthrough_feature_keys:
    chunk_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = chunk_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'chunk keys {overlap_keys} also included in passthrough keys')

  if max_length is None and sequence_length is not None:
    max_length = sequence_length[feature_key]
    if output_features[feature_key].add_eos:
      # Leave room to insert an EOS token.
      max_length -= 1
  if max_length is None:
    raise ValueError('Must specify max_length or sequence_length.')

  seed = maybe_cast_seed(seed)
  seeds = tf.unstack(tf.random.experimental.stateless_split(seed))
  tokens = features[feature_key]
  n_tokens = tf.shape(tokens)[0]
  if min_length is not None:
    length = tf.random.stateless_uniform([],
                                         minval=min_length,
                                         maxval=max_length,
                                         dtype=tf.int32,
                                         seed=seeds[0])
  else:
    length = max_length
  if uniform_random_start:
    start = tf.random.stateless_uniform(
        [],
        minval=-length + 1,  # pylint:disable=invalid-unary-operand-type
        maxval=n_tokens,
        dtype=tf.int32,
        seed=seeds[1])
    end = tf.minimum(start + length, n_tokens)
    start = tf.maximum(start, 0)
  else:
    num_segments = tf.cast(
        tf.math.ceil(
            tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32)),
        tf.int32)
    start = length * tf.random.stateless_uniform(
        [], maxval=num_segments, dtype=tf.int32, seed=seeds[1])
    end = tf.minimum(start + length, n_tokens)
  chunk = {feature_key: tokens[start:end]}
  if additional_feature_keys is not None:
    for k in additional_feature_keys:
      with tf.control_dependencies([
          tf.assert_equal(
              tf.shape(tokens)[0],
              tf.shape(features[k])[0],
              message=(f'Additional feature {k} is not the same size as '
                       f'{feature_key} along axis 0 in select_random_chunk().'))
      ]):
        chunk[k] = features[k][start:end]
  if passthrough_feature_keys is not None:
    for k in passthrough_feature_keys:
      chunk[k] = features[k]
  return chunk


def select_random_chunk(dataset: tf.data.Dataset,
                        output_features: Mapping[str, seqio.Feature],
                        max_length: Optional[int] = None,
                        feature_key: str = 'targets',
                        additional_feature_keys: Optional[Sequence[str]] = None,
                        passthrough_feature_keys: Optional[
                            Sequence[str]] = None,
                        sequence_length: Optional[Mapping[str, int]] = None,
                        uniform_random_start: bool = False,
                        min_length: Optional[int] = None,
                        **unused_kwargs) -> tf.data.Dataset:
  """SeqIO wrapper for single_example_select_random_chunk()."""

  @seqio.map_over_dataset(num_seeds=1)
  def _my_fn(x, seed):
    return single_example_select_random_chunk(
        x,
        seed,
        output_features=output_features,
        max_length=max_length,
        feature_key=feature_key,
        additional_feature_keys=additional_feature_keys,
        passthrough_feature_keys=passthrough_feature_keys,
        sequence_length=sequence_length,
        uniform_random_start=uniform_random_start,
        min_length=min_length)

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))
  return _my_fn(dataset)

def reduce_concat_tokens(dataset,
                         feature_key='targets',
                         batch_size=128,
                         **unused_kwargs):
  """Token-preprocessor to concatenate multiple unrelated documents.

  If we want to generate examples of exactly the right length,
  (to avoid wasting space on padding), then we use this function, folowed by
  split_tokens.

  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    feature_key: an string
    batch_size: an integer - how many documents to concatenate into one

  Returns:
    a dataset
  """
  dataset = dataset.map(
      lambda x: {feature_key: x[feature_key]}, num_parallel_calls=AUTOTUNE)
  dataset = dataset.padded_batch(batch_size, padded_shapes={feature_key: [-1]})

  def _my_fn(x):
    tokens = tf.reshape(x[feature_key], [-1])
    # strip padding
    tokens = tf.boolean_mask(tokens, tf.cast(tokens, tf.bool))
    return {feature_key: tokens}

  return dataset.map(_my_fn, num_parallel_calls=AUTOTUNE)


def split_tokens(dataset: tf.data.Dataset,
                 min_tokens_per_segment: Optional[int] = None,
                 max_tokens_per_segment: int = gin.REQUIRED,
                 feature_key: str = 'targets',
                 additional_feature_keys: Optional[Sequence[str]] = None,
                 passthrough_feature_keys: Optional[Sequence[str]] = None,
                 **unused_kwargs) -> tf.data.Dataset:
  """Split examples into multiple examples each.

  The intended use case is to break up long examples for use in unsupervised
  transfer-learning.

  This function is generally preceded by select_random_chunk.

  If min_tokens_per_segment is provided, the segment length is chosen randomly
  per document from a log-uniform distribution.  If min_tokens_per_segment is
  None, then the segment length is max_tokens_per_segment (except for a possibly
  shorter last segment in each document).

  Args:
    dataset: a tf.data.Dataset with dictionaries containing the key feature_key.
    min_tokens_per_segment: an optional integer
    max_tokens_per_segment: an integer, the maximum number of tokens in each
      segment. Only the final segment may be shorter.
    feature_key: a string, the feature to split
    additional_feature_keys: Additional features to split. The same chunk size
      will be used, so they should be the same size as feature_key.
    passthrough_feature_keys: Features to pass through without any splitting.

  Returns:
    a dataset
  """
  if passthrough_feature_keys:
    split_keys = set([feature_key] + (additional_feature_keys or []))
    overlap_keys = split_keys & set(passthrough_feature_keys)
    if overlap_keys:
      raise ValueError(
          f'split keys {overlap_keys} also included in passthrough keys')

  @seqio.map_over_dataset(num_seeds=1)
  def _split_tokens(x, seed):
    """Split one token sequence into multiple sequences."""
    tokens = x[feature_key]
    n_tokens = tf.shape(tokens)[0]
    if min_tokens_per_segment is None:
      length = max_tokens_per_segment
    else:
      # pick a length - log-uniformly distributed
      length = tf.cast(
          tf.exp(
              tf.random.stateless_uniform(
                  [],
                  minval=math.log(min_tokens_per_segment),
                  maxval=math.log(max_tokens_per_segment),
                  seed=seed
              )
          ),
          tf.int32)

    # Pad to a multiple of length, then use tf.reshape to split up the tokens
    # into num_segments segments each of the given length.
    num_segments = tf.cast(
        tf.math.ceil(
            tf.cast(n_tokens, tf.float32) / tf.cast(length, tf.float32))
        ,
        tf.int32)
    padding = num_segments * length - tf.shape(tokens)[0]
    feature_keys_to_split = [feature_key]
    orig_lengths = {}
    outputs = {}
    if additional_feature_keys is not None:
      feature_keys_to_split.extend(additional_feature_keys)
    for k in feature_keys_to_split:
      with tf.control_dependencies([
          tf.assert_equal(
              tf.shape(tokens)[0],
              tf.shape(x[k])[0],
              message=(f'Additional feature {k} is not the same size as '
                       f'{feature_key} along axis 0 in split_tokens().')
          )
      ]):
        shape = tf.shape(x[k])[1:]
        shape_list = x[k].shape[1:]
        padded = tf.pad(
            x[k],
            tf.concat([[[0, padding]],
                       tf.zeros([len(shape_list), 2], dtype=tf.int32)],
                      axis=0))
        orig_lengths[k] = tf.concat(
            [tf.repeat(length, num_segments - 1), [length - padding]], axis=0)
        outputs[k] = tf.reshape(
            padded, tf.concat([[-1, length], shape], axis=0))

    # To avoid memory issues, don't just replicate the passthrough features
    # for every segment; use tf.data to do it so the copies don't get
    # instantiated all at once.
    outputs_ds = tf.data.Dataset.from_tensor_slices(outputs)
    orig_lengths_ds = tf.data.Dataset.from_tensor_slices(orig_lengths)
    if passthrough_feature_keys:
      passthrough = {k: v for k, v in x.items()
                     if k in passthrough_feature_keys}
      passthrough_ds = tf.data.Dataset.from_tensors(passthrough).repeat(
          tf.cast(num_segments, tf.int64))
      return tf.data.Dataset.zip((outputs_ds, orig_lengths_ds, passthrough_ds))
    else:
      return tf.data.Dataset.zip((outputs_ds, orig_lengths_ds))

  def _strip_padding_and_merge_passthrough(
      inputs, orig_lengths, passthrough=None):
    output = {}
    for k, v in inputs.items():
      output[k] = v[:orig_lengths[k]]
    if passthrough:
      for k, v in passthrough.items():
        output[k] = passthrough[k]
    return output

  # Filter empty examples.
  dataset = dataset.filter(lambda x: tf.not_equal(tf.size(x[feature_key]), 0))

  dataset = _split_tokens(dataset).flat_map(lambda z: z)
  dataset = dataset.map(
      _strip_padding_and_merge_passthrough, num_parallel_calls=AUTOTUNE)

  return dataset


def full_lm(dataset, sequence_length, output_features):
  """Full language modeling objective with EOS only at document boundaries."""
  ds = dataset
  ds = select_random_chunk(ds, output_features=output_features,
                           feature_key='targets', max_length=65536)

  ds = append_eos_bos(ds, output_features)
  ds = reduce_concat_tokens(ds, feature_key='targets', batch_size=128)
  # Don't use `split_tokens_to_targets_length` since we've alrady added EOS.
  ds = split_tokens(ds, max_tokens_per_segment=sequence_length['targets'])
  return ds