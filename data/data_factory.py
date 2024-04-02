'''
Dataset factory to load data from huggingface and others. 
'''
import dataclasses
import pprint
import time
from functools import partial
import json
import base64
from multiprocessing import Pool

import h5py
import mlxu
from ml_collections.config_dict import config_dict, placeholder
from ml_collections import ConfigDict
from tqdm import tqdm, trange
import numpy as np

from datasets import load_dataset

from data.data_utils import get_default_vocabulary
from module.jax_utils import DataPartitioner, multihost_assert_equal, prepare_train_iter
import seqio
# from airio.core import DatasetProviderBase as airio_DatasetProviderBase
from jax.experimental import multihost_utils
import jax
from absl import logging


class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()
        config.seqio_dataset = SeqioDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        if config.type == 'huggingface':
            text_processor = TextProcessor(config.text_processor, tokenizer)
            return HuggingfaceDataset(
                config.huggingface_dataset, tokenizer, text_processor, **kwargs
            )
        elif config.type == 'json':
            text_processor = TextProcessor(config.text_processor, tokenizer)
            return JsonDataset(config.json_dataset, tokenizer, text_processor, **kwargs)
        elif config.type == 'seqio':
            return SeqioDataset(config.seqio_dataset, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError('DatasetFactory is a static class and should not be instantiated.')


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        config.base64_token_dtype = 'i4'
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.'
        )
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []

        if self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field.startswith('<|') and field.endswith('|>'):
                # Special tokens.
                field = field[2:-2]
                if field == 'bos':
                    token_buffer.append(self.tokenizer.bos_token_id)
                elif field == 'eos':
                    token_buffer.append(self.tokenizer.eos_token_id)
                else:
                    # Token ID specified directly.
                    token_buffer.append(int(field))
                loss_mask_buffer.append(mask)
            elif field.startswith('{') and field.endswith('}'):
                field = field[1:-1]
                # Base64 encoded raw tokens.
                tokens = np.frombuffer(
                    base64.b64decode(example[field]),
                    dtype=self.config.base64_token_dtype
                ).tolist()
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields]
                )
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, *aux


def rename_seqio_fields(ds):
    def _rename(ex):
        ex['input_tokens'] = ex.pop('decoder_input_tokens')
        ex['target_tokens'] = ex.pop('decoder_target_tokens')

        if "decoder_segment_ids" in ex:
            raise NotImplementedError("What is the right name for them?")

        if 'decoder_loss_weights' in ex:
            ex['loss_masks'] = ex.pop('decoder_loss_weights')
        return ex

    return ds.map(_rename)


class SeqioDataset(object):
    """ Seqio dataset.
    """
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.mixture_or_task_name = ''
        config.task_feature_field = ()
        config.task_feature_lengths = ()
        config.split = 'train'
        config.batch_size = 1
        config.shuffle = True
        config.num_epochs = placeholder(int)
        config.drop_remainder = True
        config.seed = None
        config.use_cached = False
        config.pack = False
        config.use_custom_packing_ops = False
        config.use_memory_cache = False
        config.trim_output_features = True
        # multi-modal input:
        config.image_idx_length = 144
        config.num_images = 10
        config.num_patches = 576
        config.num_pixels_per_patch = 14 * 14 * 3

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, feature_converter_cls,
                 mesh, passthrough_features=None):
        cfg = config
        self.config = config
        data_partitioner = DataPartitioner(mesh)
        data_layout = data_partitioner.get_data_layout(cfg.batch_size)
        shard_id = data_layout.shard_id
        num_shards = data_layout.num_shards
        if num_shards != jax.process_count():
            logging.warning(f"Using {num_shards} shards, but have {jax.process_count()} processes")
        shard_info = seqio.ShardInfo(index=shard_id, num_shards=num_shards)
        seed = cfg.seed
        
        assert len(cfg.task_feature_lengths) == len(cfg.task_feature_field)
        
        self.task_feature_lengths_dict = {k:v for k, v in zip(cfg.task_feature_field, cfg.task_feature_lengths)}
        if seed is None:
            # Use a shared timestamp across devices as the seed.
            seed = int(multihost_utils.broadcast_one_to_all(np.int32(time.time())))

        assert config.batch_size % shard_info.num_shards == 0
        batch_size = config.batch_size // shard_info.num_shards
        if isinstance(
            cfg.mixture_or_task_name,
            (seqio.DatasetProviderBase,),
        ):
            mixture_or_task = cfg.mixture_or_task_name
        else:
            mixture_or_task = seqio.get_mixture_or_task(cfg.mixture_or_task_name)
        
        if seed is not None:
            if not str(jax.devices()[0]).startswith('MOCK_TPU'):
                multihost_assert_equal(
                    np.array(seed),
                    (
                        f'`seed` is not same across hosts; {jax.process_index} has a seed'
                        f' of {seed}'
                    ),
                )
            logging.info(
                (
                    "Initializing dataset for task '%s' with a replica batch size of %d"
                    ' and a seed of %d'
                ),
                mixture_or_task.name,
                batch_size,
                seed,
            )

        in_memory_shuffle = cfg.shuffle
        if not cfg.drop_remainder:
            # Used if we want to evaluate on an eval dataset without dropping any examples.
            # To do this, we pad the dataset with dummy examples marked as invalid in their
            # metadata we can still get fixed-sized batches.
            assert cfg.num_epochs is not None
            assert not cfg.pack
            ds = mixture_or_task.get_dataset(
                sequence_length=self.task_feature_lengths_dict,
                split=cfg.split,
                shuffle=in_memory_shuffle,
                num_epochs=cfg.num_epochs,
                use_cached=cfg.use_cached,
                seed=seed,
                trim_output_features=cfg.trim_output_features,
            )

            n = len(ds)
            remainder = n % config.batch_size
            if remainder > 0:
                to_pad = config.batch_size - remainder
            else:
                to_pad = 0
            assert "metadata/valid" not in ds.element_spec
            def add_valid(x):
                x["metadata/valid"] = True
                return x
            def add_invalid(x):
                x["metadata/valid"] = False
                return x
            ds = ds.map(add_valid)
            if to_pad > 0:
                to_pad = ds.take(1).map(add_invalid).cache().repeat(to_pad)
                ds = ds.concatenate(to_pad)

            # We shard after padding to ensure shards are the same length
            ds = ds.shard(num_shards=num_shards, index=shard_id)

            ds = feature_converter_cls(
                pack=cfg.pack, use_custom_packing_ops=cfg.use_custom_packing_ops,
                passthrough_features=passthrough_features
            )(ds, task_feature_lengths=self.task_feature_lengths_dict)
            data_iter = ds.batch(batch_size, drop_remainder=True)
        else:
            data_iter = seqio.get_dataset(
                mixture_or_task_name=mixture_or_task,
                task_feature_lengths=self.task_feature_lengths_dict,
                dataset_split=cfg.split,
                shuffle=in_memory_shuffle,
                num_epochs=cfg.num_epochs,
                feature_converter=feature_converter_cls(
                    pack=cfg.pack, use_custom_packing_ops=cfg.use_custom_packing_ops,
                    passthrough_features=passthrough_features
                ),
                shard_info=shard_info,
                use_cached=cfg.use_cached,
                seed=seed,
                trim_output_features=cfg.trim_output_features,
                batch_size=batch_size,
            )
        data_iter = rename_seqio_fields(data_iter)
        self._tfds_dataset = data_iter
        
        self._dataset = prepare_train_iter(
            data_iter,
            partitioner=data_partitioner,
            data_layout=data_layout,
        )

    def reset(self):
        self._dataset.reset()

    def __iter__(self):
        return ((x, {}) for x in self._dataset)

    @property
    def seq_length(self):
        return self.task_feature_lengths_dict['targets']

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size
    
    @property
    def image_idx_length(self):
        return self.config.image_idx_length

    @property
    def num_patches(self):
        return self.config.num_patches

    @property
    def num_images(self):
        return self.config.num_images

    @property
    def num_pixels_per_patch(self):
        return self.config.num_pixels_per_patch
    
    def get_state_dict(self):
        return dict(config=self.config)


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.batch_token_dtype = 'i4'

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(
            self.config.path, name, split=split, streaming=self.config.streaming
        )

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    batch = {
                        'input_tokens': np.array(token_buffer[:chunk_size], dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=self.config.batch_token_dtype).reshape(
                            self.config.batch_size, -1
                        ),
                        'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                            self.config.batch_size, -1
                        ),
                    }
                    if self.config.always_start_with_bos:
                        batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:   # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(), self.config.tokenizer_parallel_batch_size
            )
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn, next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size
                )
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn, next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size
                    )
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index in self.parallel_example_iterator():
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = (
                    (self._total_tokens - start_tokens) / (time.time() - start_time)
                )
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens': np.array(token_buffer[:chunk_size], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'target_tokens': np.array(token_buffer[1:chunk_size + 1], dtype=np.int32).reshape(
                        self.config.batch_size, -1
                    ),
                    'loss_masks': np.array(loss_mask_buffer[1:chunk_size + 1], dtype=np.float32).reshape(
                        self.config.batch_size, -1
                    ),
                }
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))
        self._index = state_dict.get('index', self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens', self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def image_idx_length(self):
        return self.config.image_idx_length

    @property
    def num_patches(self):
        return self.config.num_patches

    @property
    def num_images(self):
        return self.config.num_images

    @property
    def num_pixels_per_patch(self):
        return self.config.num_pixels_per_patch

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)
