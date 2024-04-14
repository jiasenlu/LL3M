import tensorflow as tf
from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple, Union

import seqio
from data.tasks import TaskRegistry
from data.mixtures import MixtureRegistry
from data.data_utils import get_default_vocabulary, MultiModalLMFeatureConverter, LMFeatureConverter

from seqio import test_utils
import numpy as np
import gin

# print(tf.executing_eagerly())
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

def test_multimodal_packing():
    def create_default_dataset(
        x: Sequence[Mapping[str, Sequence[int]]],
        feature_names: Sequence[str] = ("inputs", "targets"),
        output_types: Optional[Mapping[str, tf.dtypes.DType]] = None,
        output_shapes: Optional[Mapping[str, Tuple[None]]] = None,
    ) -> tf.data.Dataset:
        """Creates a dataset from the given sequence."""
        if output_types is None:
            output_types = {feature_name: tf.int32 for feature_name in feature_names}
        if output_shapes is None:
            output_shapes = {feature_name: [None] for feature_name in feature_names}

        ds = tf.data.Dataset.from_generator(
            lambda: x, output_types=output_types, output_shapes=output_shapes
        )
        return ds

    image_input_idx_1 = [[0, 1, 2]]
    image_input_idx_2 = [[0, 1, 2], [4, 5, 6]]
    image_input_idx_3 = [[0, 1, 2]]

    x = [
        {
            "targets": [0, 0, 0, 3, 9, 1], 
            "decoder_loss_weights": [0, 0, 0, 1, 1, 1],
            "images": np.ones([1, 3, 2], dtype=np.float32), 
            "image_input_idx": image_input_idx_1},
        {
            "targets": [0, 0, 0, 4, 0, 0, 0, 1], 
            "decoder_loss_weights": [0, 0, 0, 1, 1, 1, 1, 1],
            "images": np.ones([2, 3, 2],dtype=np.float32)+1, 
            "image_input_idx": image_input_idx_2},
        {
            "targets": [0, 0, 0, 0, 2, 3, 4, 5], 
            "decoder_loss_weights": [0, 0, 0, 1, 1, 1, 1, 1],
            "images": np.ones([1, 3, 2], dtype=np.float32)+2, 
            "image_input_idx":image_input_idx_3},]

    ds = create_default_dataset(
    x, 
    feature_names = ("targets", "images", "image_input_idx", "decoder_loss_weights"), 
    output_types = {'targets': tf.int32, 'images': tf.float32, "image_input_idx": tf.int32, "decoder_loss_weights": tf.int32},
    output_shapes = {'targets': [None], 'images': [None, 3, 2],  "image_input_idx": [None, 3], "decoder_loss_weights": [None]}
    )

    task_feature_lengths = {"targets": 30, "images": 4, "image_input_idx": 4, "decoder_loss_weights": 30}

    converter = MultiModalLMFeatureConverter(pack=True)

    converted_ds = converter(ds, task_feature_lengths)

    for i, ex in zip(range(100000), converted_ds.as_numpy_iterator()):
        pass

def test_multimodal_datasets():
    gin_config = ''
    gin_bindings = 'data.data_utils.get_default_vocabulary.tokenizer_type = "llama", data.data_utils.get_default_vocabulary.has_extra_token = True'
    gin_bindings = gin_bindings.split(',')
    
    gin.parse_config_files_and_bindings(config_files=gin_config, bindings=gin_bindings)

    seq_len = {
        "targets": 4096,
        "decoder_loss_weights": 4096, 
        "images": 5, 
        "image_positions": 5,
        "image_input_idx": 5,
    }

    dataset = seqio.get_mixture_or_task("llava_v1_5_mix665k").get_dataset(
        sequence_length=seq_len,
        split="train",
        num_epochs=1,
        shard_info=seqio.ShardInfo(index=0, num_shards=1),
        use_cached=False,
        seed=32,
        shuffle=False,
    )

    converter = MultiModalLMFeatureConverter(pack=False, use_custom_packing_ops=False)
    dataset = converter(dataset, seq_len)
    vocab = get_default_vocabulary()
    # dataset = dataset.batch(8, drop_remainder=True)

    for i, ex in zip(range(100000), dataset.as_numpy_iterator()):
        # token = ex['decoder_input_tokens'] * np.array(ex['decoder_input_tokens'] != -1, np.int32)
        # print(vocab.decode_tf(token[0]))
        print(ex['images'].shape)
        import pdb; pdb.set_trace()
        
def test_nlp_datasets():
    seq_len = {
        "targets": 2048,
        # "decoder_loss_weights": 2048, 
    }

    dataset = seqio.get_mixture_or_task("c4").get_dataset(
        sequence_length=seq_len,
        split="train",
        num_epochs=1,
        shard_info=seqio.ShardInfo(index=0, num_shards=1),
        use_cached=False,
        seed=32,
        shuffle=False,
    )

    vocab = get_default_vocabulary('gemma')

    converter = LMFeatureConverter(pack=False, use_custom_packing_ops=False, bos_id=vocab.bos_token_id)
    dataset = converter(dataset, seq_len)
    for i, ex in zip(range(100000), dataset.as_numpy_iterator()): 
        vocab.decode(ex['decoder_input_tokens'])
        import pdb; pdb.set_trace()
    
    

test_multimodal_datasets()