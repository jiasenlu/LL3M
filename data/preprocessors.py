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


# ------------------------
# Multimodal Preprocessor
# ------------------------

@seqio.map_over_dataset
def extract_llava(ex, sequence_length, output_features):
    tf.assert_equal(tf.shape(ex['conversations']['value'])[0], 2)
    prompt = ex['conversations']['value'][0]
    text = ex['conversations']['value'][1]
    ex.pop('conversations')
    ex["text"] = text
    ex["prompt"] = prompt
    return ex
  
def select_tiling(h, w, patch_size, max_num_patches):
    """Decide how best to divide in image of size [w, h] in up to max_num_patches of size patch_size"""
    original_size = tf.stack([h, w])  # [1, 2]
    original_res = h * w
    tilings = []
    for i in range(1, max_num_patches+1):
        for j in range(1, max_num_patches+1):
            if i*j <= max_num_patches:
                tilings.append((i, j))
    # sort so argmin and argmax favour smaller tilings in the event of a tie
    tilings.sort(key=lambda x: (x[0]*x[1], x[0]))
    candidate_tilings = tf.constant(tilings, dtype=tf.int32)  # [n_resolutions, 2]
    candidate_resolutions = candidate_tilings * patch_size  # [n_resolutions, 2]

    # How much we would need to scale the image to fit exactly in each tiling
    required_scale_d = tf.cast(candidate_resolutions, tf.float32) / tf.cast(original_size[None, :], tf.float32)
    required_scale = tf.reduce_min(required_scale_d, axis=-1, keepdims=True)  # [n_resolutions, 1]
    if tf.reduce_all(required_scale < 1):
        # We are forced to downscale, so try to minimize the amount of downscaling
        ix = tf.argmax(required_scale)[0]
    else:
        # Pick the resolution that required the least upscaling so that it most closely fits the image
        required_scale = tf.where(required_scale < 1.0, 10e9, required_scale)
        ix = tf.argmin(required_scale)[0]
    return candidate_tilings[ix]
  
@gin.configurable()
def image_to_patches_and_tokens(
    image, is_training,
    mode="patchify-v2-and-resize-c2",
    do_random_scale=True,
    base_image_input_size=BASE_IMAGE_INPUT_SIZE,
    base_image_input_d=BASE_IMAGE_INPUT_D,
    max_num_patches=MAX_NUM_PATCHES,
    random_scale_max=RANDOM_SCALE_MAX,
    random_scale_min=RANDOM_SCALE_MIN,
    random_scale_ratio=RANDOM_SCALE_RATIO,
    image_token_length_w=12,
    image_token_length_h=12,
    use_col_tokens=True, 
    use_img_start_end_token=True
):
    """
    Args:
        image: [w, h, 3] image to patchify
    Returns:
        tokens: (n_tokens,) tf.int32 tokens, pad tokens indicate where to insert the image features
                            there will exactly `IMAGE_TOKEN_LENGTH` pad tokens
        patches: (n_patches, n_subpatches, subpatch_dim) individual patches, `n_patches` might
                 change between images but the other dimension are fixed
    """
    if image.dtype == tf.string:
        image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if do_random_scale:
        do_random_scale = is_training

    tokens_per_image = image_token_length_w * image_token_length_h
    image_base_patch_w = base_image_input_size[1] // base_image_input_d
    image_base_patch_h = base_image_input_size[0] // base_image_input_d
    extra_image = False
    patch_ordering = None

    def _resize(_image, sz):
        return resize_and_pad(
            _image, sz,
            do_random_scale=do_random_scale,
            random_scale_max=random_scale_max,
            random_scale_min=random_scale_min,
            shrink_both_sides=True,
            do_flip_if_vertical=False,
            random_scale_ratio=random_scale_ratio,
            resize_method='random' if is_training else tf.image.ResizeMethod.BILINEAR)

    if mode == "resize":
        patches, img_mask, this_image_info = _resize(image, base_image_input_size)
        image_layout_impatch_w = 1
        image_layout_impatch_h = 1
        patches = einops.rearrange(
            patches, '(dy h dh) (dx w dw) c -> (dy dx) (h w) (dh dw c)',
            dh=base_image_input_d,
            dw=base_image_input_d,
            dy=1,
            dx=1,
            h=image_base_patch_h,
            w=image_base_patch_w
        )
        patch_ordering = tf.range(tokens_per_image)[None, :]

    elif mode in ["patchify", "patchify-and-resize", "patchify-v2", "patchify-v2-and-resize", "patchify-v2-and-resize-c2"]:
        original_image_w = tf.shape(image, out_type=tf.int32)[0]
        original_image_h = tf.shape(image, out_type=tf.int32)[1]
        assert base_image_input_size[0] == base_image_input_size[1]
        base_patch_size = base_image_input_size[0]
        tiling = select_tiling(original_image_w, original_image_h, base_patch_size, max_num_patches)

        patches, img_mask, this_image_info = _resize(
            image, [tiling[0]*base_patch_size, tiling[1]*base_patch_size])
        patches = einops.rearrange(
            patches, '(dy h dh) (dx w dw) c -> (dy dx) (h w) (dh dw c)',
            dh=base_image_input_d,
            dw=base_image_input_d,
            dy=tiling[0],
            dx=tiling[1],
            h=image_base_patch_h,
            w=image_base_patch_w
        )
        if 'v2' in mode:
            # Order patches left-to-right not crop-by-crop
            patch_ordering = tf.reshape(
                tf.range(tokens_per_image*tiling[0]*tiling[1]),
                [tiling[0], tiling[1], image_token_length_w, image_token_length_h])
            patch_ordering = tf.transpose(patch_ordering, [0, 2, 1, 3])
            patch_ordering = tf.reshape(patch_ordering, (-1, tokens_per_image))
        else:
            patch_ordering = None

        # given image size, determine the number of patch size.
        image_layout_impatch_w = tiling[0]
        image_layout_impatch_h = tiling[1]

        if "resize" in mode:
            extra_image = True
            resized = _resize(image, base_image_input_size)[0]
            resized = einops.rearrange(
                resized, '(dy h dh) (dx w dw) c -> (dy dx) (h w) (dh dw c)',
                dh=base_image_input_d,
                dw=base_image_input_d,
                dy=1,
                dx=1,
                h=image_base_patch_h,
                w=image_base_patch_w
            )
            if 'c2' in mode:
              patches = tf.concat([resized, patches], 0)              
            else:
              patches = tf.concat([patches, resized], 0)
              
            if patch_ordering is not None:
                patch_ordering = tf.concat(
                  [tf.range(0, tokens_per_image)[None, :], patch_ordering+tokens_per_image], 0)
    else:
        raise NotImplementedError(mode)
    
    if use_img_start_end_token or use_col_tokens:
      special_token_ids = get_special_token_ids()
      image_patch_token = special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]
      image_start_token = special_token_ids[DEFAULT_IM_START_TOKEN]
      image_end_token = special_token_ids[DEFAULT_IM_END_TOKEN]
      image_col_token = special_token_ids[DEFAULT_IM_COL_TOKEN]
    else:
      image_patch_token = 0
      
    per_row = tf.fill((image_token_length_w*image_layout_impatch_w,), image_patch_token,)
    if use_col_tokens:
        per_row = tf.concat([per_row, [image_patch_token]], 0)

    joint = tf.tile(per_row, [image_token_length_h * image_layout_impatch_h])
    
    if use_img_start_end_token:
      joint = [
          [image_start_token],
          joint,
          [image_end_token]
      ]
      
    if extra_image:
        per_row = tf.fill((image_token_length_w,), image_patch_token,)
        if use_col_tokens:
            per_row = tf.concat([per_row, [image_col_token]], 0)
        extra_tokens = tf.tile(per_row, [image_token_length_h])
        if 'c2' in mode:
          if use_img_start_end_token:
            joint = [
                [image_start_token],
                extra_tokens,
                [image_end_token],
            ] + joint
          else:
            joint = [extra_tokens] + joint
        else:
          if use_img_start_end_token:
            joint += [
                [image_start_token],
                extra_tokens,
                [image_end_token]
            ]
          else:
            joint += [extra_tokens]
          
    return patches, tf.concat(joint, 0), (image_layout_impatch_w, image_layout_impatch_w), patch_ordering

@gin.configurable()
def multimodal_preprocessor(
  ds, sequence_length, output_features,
  flatten_by_caption=False,
  include_response=True,
  prompt_type = 'mistral',
  include_metadata=False,
  decode_jpeg = False,
  image_token_length_w = 12,
  image_token_length_h = 12,
  use_col_tokens = True,
  use_img_start_end_token = True,
):
  vocab = get_default_vocabulary()
  is_training = sequence_length.get('is_training', True)

  def to_inputs_and_targets(ex, seeds=None):

    if decode_jpeg:
      image = tf.image.decode_jpeg(ex['image'], channels=3)
    else:
      image = ex['image']
      
    image, image_tokens, _, patch_order = image_to_patches_and_tokens(
        image, is_training, 
        image_token_length_w=image_token_length_w, 
        image_token_length_h=image_token_length_h, 
        use_col_tokens=use_col_tokens, 
        use_img_start_end_token=use_img_start_end_token)

    if prompt_type == "plain-v1":
        prompt_list = tf.constant([
            'Here is an image\n',
            'Picture:\n',
            'Image:\n',
            'Examine this image:\n',
            'Look at this image:\n',
        ])
        index = tf.random.uniform(shape=(), minval=0, maxval=len(prompt_list), dtype=tf.int32)
        encoded_prefix = tf.concat([vocab.encode_tf(prompt_list[index]), image_tokens], 0)
    elif prompt_type == "null":
        encoded_prefix = image_tokens
    elif prompt_type == "mistral":
        prompt_template = PROPMPT_MANAGER[prompt_type]
        prompt_list = tf.constant(PROMPT_LLAVA_PRETRAIN)
        index = tf.random.uniform(shape=(), minval=0, maxval=len(prompt_list), dtype=tf.int32)
        
        image_token_strings = vocab.decode_tf(image_tokens)
        # remove the system prompt and add the caption prompt.       
        image_token_string_with_prompt = tf.strings.regex_replace(prompt_list[index], '<image>', image_token_strings)
        
        encoded_prefix = tf.concat([
            vocab.encode_tf(prompt_template['B_INST']),
            vocab.encode_tf(image_token_string_with_prompt),
            vocab.encode_tf(prompt_template['E_INST']),
        ], 0)
    elif prompt_type == "llava":
        import pdb; pdb.set_trace()
    else:
        prompt_template = PROPMPT_MANAGER[prompt_type]
        prompt_list = tf.constant(PROMPT_LLAVA_PRETRAIN)
        index = tf.random.uniform(shape=(), minval=0, maxval=len(prompt_list), dtype=tf.int32)
        after_image = prompt_template['E_INST']
        encoded_prefix = tf.concat([
            vocab.encode_tf(tf.strings.join([prompt_template['B_INST'], prompt_template['SYS_PREFIX']])),
            image_tokens,
            vocab.encode_tf(prompt_template['E_INST']),
        ], 0)

    prefix_loss_weights = tf.zeros(tf.shape(encoded_prefix), tf.int32)

    if include_response:
        if len(ex['text'].shape) > 0:
            index = tf.random.uniform(shape=(), minval=0, maxval=tf.shape(ex['text'])[0], dtype=tf.int32)
            text = ex['text'][index]
        else:
            text = ex['text']
        encoded_response = vocab.encode_tf(text)
        encoded_response = tf.pad(encoded_response, [[0, 1]], constant_values=vocab.eos_token_id)
        response_loss_weights = tf.ones(tf.shape(encoded_response), tf.int32)
        targets = tf.concat([encoded_prefix, encoded_response], axis=0)
        decoder_loss_weights = tf.concat([prefix_loss_weights, response_loss_weights], axis=0)
    else:
        # Only the prompt, used for evaluation
        targets = encoded_prefix
        decoder_loss_weights = prefix_loss_weights
    
    if use_img_start_end_token:
      image_patch_token = get_special_token_ids()[DEFAULT_IMAGE_PATCH_TOKEN]
    else:
      image_patch_token = 0
      
    image_input_idx = targets == image_patch_token
    image_input_idx = tf.experimental.numpy.nonzero(image_input_idx)[0]

    if patch_order is not None:
        # patch_order is patch_ix->order
        # First we build an array of patch_ix's sorted by their ordering
        patch_order = tf.reshape(patch_order, [-1])
        n = tf.shape(patch_order)[0]
        sorted_patch_ixs = tf.scatter_nd(patch_order[:, None], tf.range(n), [n])
        # Now `image_input_idx` so it points to lists tokens in order patches should
        # be inserted instead of sequentially
        image_input_idx = tf.gather(image_input_idx, sorted_patch_ixs)

    image_input_idx = tf.reshape(image_input_idx, [-1, image_token_length_w * image_token_length_h])
    image_input_idx = tf.cast(image_input_idx, tf.int32)
    out = {
        'targets': targets,
        'image_input_idx': image_input_idx,
        'images': image,
        'decoder_loss_weights': decoder_loss_weights,
    }
    if include_metadata:
        if len(ex["text"].shape) > 0:
            # FIXME can this be variable lengths after all?
            out["metadata/captions"] = tf.strings.reduce_join(
                tf.strings.regex_replace(ex['text'], "\\s+", " "),
                separator="\n"
            )
        else:
            out["metadata/captions"] = ex["text"]
        if "url" in ex:
            out["metadata/image_url"] = ex["url"]
        if "image/filename" in ex:
            image_id = tf.strings.substr(ex["image/filename"], 0, tf.strings.length(ex["image/filename"])-4)
            out["metadata/image_id"] = tf.strings.to_number(image_id)
        out["metadata/image"] = resize_and_pad(
            tf.image.convert_image_dtype(ex["image"], dtype=tf.float32), (336, 336),
            do_random_scale=False, normalize=False, do_flip_if_vertical=False)[0]
    return out

  if len(ds.element_spec["text"].shape) > 0:
      if flatten_by_caption:
          ds = flatten_parts(ds, parts=["text"])

  ds = ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return ds