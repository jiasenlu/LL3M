'''
From: 
'''

import os
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Sequence, Iterable
import json
import tempfile
from functools import partial
from absl import logging

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as PS
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.linen import partitioning as nn_partitioning
from flax.linen.dtypes import promote_dtype
import einops

import sentencepiece as spm
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward

from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from mlxu import function_args_to_config, load_pickle, open_file

from data.transformer_tokenizer import LLaMATokenizer
from module.jax_utils import (
    with_sharding_constraint, get_jax_mesh
)

from models.openLLM.model import FlaxOpenLLMBlockCollection, RMSNorm, FlaxOpenLLMMLP
from models.llava.vit import VisionTransformer, MultiHeadDotProductAttention, VIT_STANDARD_CONFIGS

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

LLAVA_STANDARD_CONFIGS = {
    "llava-v1.5-vicuna-7b": {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 2,
        'num_attention_heads': 32,
        'num_key_value_heads': 32,
        'max_sequence_length': 4096,
        'max_position_embeddings': 8192,
        'rope_theta': 10000.0, 
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'hidden_act': 'silu', 
        'z_loss': 0.001,
        'norm_module': 'RMSNorm',
        "mm_vision_tower": "ViT-L/14-336",
    },
    "llava-v1.6-vicuna-7b": {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 2,
        'num_attention_heads': 32,
        'num_key_value_heads': 32,
        'max_sequence_length': 4096,
        'max_position_embeddings': 8192,
        'rope_theta': 10000.0, 
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'hidden_act': 'silu', 
        'z_loss': 0.001,
        'norm_module': 'RMSNorm',
        "mm_vision_tower": "ViT-L/14-336",
    },
    "llava-v1.6-vicuna-7b-flash": {
        'vocab_size': 32000,
        'hidden_size': 4096,
        'intermediate_size': 11008,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'num_key_value_heads': 32,
        'max_sequence_length': 4096,
        'max_position_embeddings': 8192,
        'rope_theta': 10000.0, 
        'initializer_range': 0.02,
        'rms_norm_eps': 1e-5,
        'use_cache': True,
        'tie_word_embeddings': False,
        'hidden_act': 'silu', 
        'z_loss': 0.001,
        'norm_module': 'RMSNorm',
        "mm_vision_tower": "ViT-L/14-336",
        'scan_attention': True,
        'scan_mlp': True,
    },
}

# Type annotations
Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Sequence[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)

class LlavaConfig(PretrainedConfig):
    r"""
    ```python
    >>> from transformers import LLaMAModel, LlavaConfig
    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlavaConfig()
    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LLaMAModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llava"

    def __init__(
        self,
        additional_vocab_size=128,
        use_cache=True,
        # pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=False,
        remat_block='',
        remat_attention='',
        remat_mlp='',
        scan_attention=False,
        scan_mlp=False,
        scan_query_chunk_size=1024,
        scan_key_chunk_size=1024,
        scan_mlp_chunk_size=1024,
        fcm_min_ratio=0.0,
        fcm_max_ratio=0.0,
        normalize_input_embeds=False,
        **kwargs,
    ):
        self.additional_vocab_size = additional_vocab_size
        self.use_cache = use_cache
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.remat_block = remat_block
        self.remat_attention = remat_attention
        self.remat_mlp = remat_mlp
        self.scan_attention = scan_attention
        self.scan_mlp = scan_mlp
        self.scan_query_chunk_size = scan_query_chunk_size
        self.scan_key_chunk_size = scan_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.fcm_min_ratio = fcm_min_ratio
        self.fcm_max_ratio = fcm_max_ratio
        self.normalize_input_embeds = normalize_input_embeds

        super().__init__(
            # pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config
    
    @staticmethod
    def get_jax_mesh(axis_dims):
        return get_jax_mesh(axis_dims, ('dp', 'fsdp', 'mp'))
    
    @staticmethod
    def get_partition_rules():
        """ Parition rules for GPTJ. Note that these rules are orderd, so that
            the beginning rules match first. It is important to use
            PartitionSpec() instead of None here because JAX does not treat
            None as a pytree leaf.
        """
        return (
            # vision part:   
            
            ("patch_embedding/kernel", PS("mp", "fsdp")),
                     
            ("attention/(wq|wk|wv)/bias", PS("mp",)),
            ("attention/wo/bias", PS("fsdp",)),
            
            ("feed_forward/w1/bias", PS("mp",)),
            ("feed_forward/w2/bias", PS("fsdp",)),
            
            ("mm_projector/w1/kernel", PS("fsdp", "mp")),
            ("mm_projector/w2/kernel", PS("mp", "fsdp")),

            # embeddings
            ("transformer/wte/embedding", PS("mp", "fsdp")),
            # ("transformer/wte/new_embedding", PS("mp", "fsdp")),
            # atention
            ("attention/(wq|wk|wv)/kernel", PS("fsdp", "mp")),
            ("attention/wo/kernel", PS("mp", "fsdp")),
            # mlp
            ("feed_forward/w1/kernel", PS("fsdp", "mp")),
            ("feed_forward/w2/kernel", PS("mp", "fsdp")),
            ("feed_forward/w3/kernel", PS("fsdp", "mp")),
            # layer norms
            ("attention_norm/kernel", PS(None)),
            ("ffn_norm/kernel", PS(None)),
            # output head
            ("transformer/ln_f/kernel", PS(None)),
            ("lm_head/kernel", PS("fsdp", "mp")),
            ('.*', PS(None)),
        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple(['transformer/wte/embedding', "transformer/wte/new_embedding", 'attention_norm/kernel', 'ffn_norm/kernel'])

    @staticmethod
    def get_trainable_params():
        return tuple(['mm_projector', 'transformer/wte/new_embedding'])
        # return tuple([r'^.*'])

    @staticmethod
    def rng_keys():
        return ('params', 'dropout', 'fcm')

    @staticmethod
    def get_tokenizer_config(updates=None):
        config = ConfigDict()
        config.vocab_file = 'data/tokenizer.model'
        config.add_bos_token = False
        config.add_eos_token = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def get_tokenizer(cls, config, padding_side='left', truncation_side='right'):
        config = cls.get_tokenizer_config(config)
        assert config.vocab_file != '', 'vocab_file must be specified'
        tokenizer = LLaMATokenizer(
            vocab_file=config.vocab_file,
            add_bos_token=config.add_bos_token,
            add_eos_token=config.add_eos_token,
            padding_side=padding_side,
            truncation_side=truncation_side,
        )
        return tokenizer

    @classmethod
    def load_config(cls, path):
        if path in LLAVA_STANDARD_CONFIGS:
            config = LLAVA_STANDARD_CONFIGS[path]
            return cls.from_dict(config | VIT_STANDARD_CONFIGS[config['mm_vision_tower']])
        load_type, load_path = path.split('::', 1)
        if load_type == 'pickle':
            return cls.from_dict(load_pickle(load_path)['llama_config'])
        elif load_type == 'json':
            with open_file(load_path, 'r') as fin:
                raw_config = fin.read()
            return cls.from_dict(json.loads(raw_config))
        else:
            raise ValueError(f'Unsupported load config type: {load_type}')

remat = nn_partitioning.remat


def QuickGELU(x): return x * nn.sigmoid(1.702 * x)
    
class MLP(nn.Module):
    mlp_dims: DType = int
    out_dims: DType = int
    dtype: DType = jnp.float32
    param_dtype: DType = jnp.float32
    
    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(
            self.mlp_dims,
            dtype=self.dtype,
            use_bias=True,
            name='w1',
        )(x)
        x = jax.nn.gelu(x, approximate=False)
        x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))
        x = nn.Dense(
            self.out_dims,
            dtype=self.dtype,
            use_bias=True,
            name='w2',
        )(x)

        return x
    
class FlaxLlavaPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LlavaConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: LlavaConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.hidden_size,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    @add_start_docstrings_to_model_forward("")
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        images=None,
        image_input_idx=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        decoding_with_cache: Optional[bool] = None,
        append_last_valid_logits: bool=None,

    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        batch_size, sequence_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxGPTJAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        extra_args = dict()
        if images is not None:
            extra_args["images"] = images
            extra_args["image_input_idx"] = image_input_idx

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            deterministic=not train,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rngs=rngs,
            mutable=mutable,
            decoding_with_cache=decoding_with_cache is not None,
            append_last_valid_logits=append_last_valid_logits,
            **extra_args
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs

class FlaxLlavaModule(nn.Module):
    config: LlavaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        
        self.image_vit = VisionTransformer(self.config, name='image_vit')
        mlp_module = MLP
        
        self.mm_projector = mlp_module(
            self.config.hidden_size,
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
        self.embed_dim = self.config.hidden_size
        self.wte = nn.Embed(
            self.config.vocab_size + self.config.additional_vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        self.h = FlaxOpenLLMBlockCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision)
        self.ln_f = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=self.dtype, param_dtype=self.param_dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        images,
        image_input_idx,    
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        decoding_with_cache: Optional[bool] = None
    ):
        batch_size, seq_length = input_ids.shape
        input_embeds = self.wte(input_ids.astype("i4"))        
        
        
        if images is not None and not decoding_with_cache:  
            _, num_image, num_patch, _ = images.shape

            image_features = self.image_vit(images)
            image_features = jax.lax.stop_gradient(image_features)

            # MLP layer to map the feature.
            image_features = self.mm_projector(image_features)

            # inster the image feature into the embedding.
            image_features = jnp.reshape(image_features, (batch_size, num_image*num_patch, -1))
            image_input_idx = jnp.reshape(image_input_idx, (batch_size, num_image*num_patch))
        
            input_embeds = input_embeds.at[jnp.arange(batch_size)[:, None], image_input_idx].add(image_features)
            
        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )

@add_start_docstrings("", "")
class FlaxLlavaModel(FlaxLlavaPreTrainedModel):
    module_class = FlaxLlavaModule

class FlaxLlavaForCausalLMModule(nn.Module):
    config: LlavaConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32
    precision: Optional[Union[jax.lax.Precision, str]]=None

    def setup(self):
        self.transformer = FlaxLlavaModule(self.config, dtype=self.dtype)

        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            precision=self.precision,
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        segment_ids: Optional[jnp.ndarray] = None,
        images: Optional[jnp.ndarray] = None,
        image_input_idx: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        decoding_with_cache: bool=False,
        append_last_valid_logits=None
    ):
        
        batch_size, seq_length = input_ids.shape
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.clip(jnp.cumsum(attention_mask, axis=-1) - 1, a_min=0),
                (batch_size, seq_length)
            )
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            images,
            image_input_idx,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if append_last_valid_logits is not None:
            logging.info("Append logits")
            last_valid_logit = lm_logits[jnp.arange(lm_logits.shape[0]), append_last_valid_logits]
            lm_logits = jnp.concatenate([lm_logits[:, :-1], last_valid_logit[:, None]], axis=1)
            
        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

@add_start_docstrings("", "")
class FlaxLlavaForCausalLM(FlaxLlavaPreTrainedModel):
    module_class = FlaxLlavaForCausalLMModule

    def prepare_inputs_for_generation(
        self, input_ids, max_length,
        images=None, image_input_idx=None,
        attention_mask: Optional[jax.Array] = None):

        if attention_mask is None:
            attention_mask = (input_ids != -1).astype(jnp.int32)

        # This would shift from left padding to right padding, which is what we want for inference,
        # but doing that seems to break the LLM even though it shouldn't in theory....
        # if attention_mask is not None:
            # bs, seq_len = attention_mask.shape
            # n_to_shift = jnp.sum(attention_mask == 0, axis=1)
            # # n_to_shift = n_to_shift * jnp.array([1]+[0]*7, dtype=n_to_shift.dtype)
            # if images is not None:
            #     image_input_idx = image_input_idx + n_to_shift[:, None, None]
            #
            # def _shift_right(_val):
            #     shifted = []
            #     for i in range(bs):
            #         shifted.append(jnp.roll(_val[i], n_to_shift[i]))
            #     return jnp.stack(shifted, 0)
            # input_ids = _shift_right(input_ids)
            # attention_mask = _shift_right(attention_mask)
        #
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since GPTJ uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # Max so right padding tokens have a valid position id
            position_ids = jnp.maximum(attention_mask.cumsum(axis=-1) - 1, 0)
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        out = {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
            "append_last_valid_logits": attention_mask.sum(axis=-1) - 1
        }
        if images is not None:
            out["images"] = images
            out["image_input_idx"] = image_input_idx
        return out

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        model_kwargs["decoding_with_cache"] = True
        if "append_last_valid_logits" in model_kwargs:
            del model_kwargs["append_last_valid_logits"]
        if "images" in model_kwargs:
            del model_kwargs["images"]
            del model_kwargs["image_input_idx"]
        return model_kwargs
