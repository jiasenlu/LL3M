import logging
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import math
import einops

import jax
from jax import lax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax.numpy as jnp
import functools
from flax import struct

from jax.sharding import PartitionSpec as PS
from flax.linen.attention import dot_product_attention_weights
from flax.linen import partitioning as nn_partitioning

from transformers.configuration_utils import PretrainedConfig
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from mlxu import function_args_to_config, load_pickle, open_file

import tempfile
from tqdm import tqdm
import requests
import os
import random

from module.jax_utils import (
        with_sharding_constraint, get_gradient_checkpoint_policy, get_jax_mesh
)
from module.bpt import blockwise_ffn, blockwise_attn

# from .config import PARAMTER_DTYPE

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]

Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_kernel_init = nn.initializers.glorot_uniform()

remat = nn_partitioning.remat


VIT_STANDARD_CONFIGS = {
    'ViT-L/14-336': {
        'image_patch_size': 14,
        'image_pos_patch_size': 14,
        'image_emb_dim': 1024,
        'image_num_heads': 16,
        'image_num_layers': 23,
        'image_head_dim': 64,
        'image_mlp_dim': 4096,
        'image_mlp_activations': ('gelu',),
        'image_dropout_rate': 0.0,
        'image_num_pos': 577,
        'image_default_input_size': (336, 336),
        'image_pooling_h': 2,
        'image_pooling_w': 2,
        'image_num_patch': (24, 24),
        'image_norm_eps': 1e-5,
        'image_num_key_value_heads': 16
    },
    'debug': {
        'image_patch_size': 14,
        'image_pos_patch_size': 14,
        'image_emb_dim': 1024,
        'image_num_heads': 16,
        'image_num_layers': 2,
        'image_head_dim': 64,
        'image_mlp_dim': 4096,
        'image_mlp_activations': ('gelu',),
        'image_dropout_rate': 0.0,
        'image_num_pos': 577,
        'image_default_input_size': (336, 336),
        'image_pooling_h': 2,
        'image_pooling_w': 2,
        'image_num_patch': (24, 24),
        'image_norm_eps': 1e-5,
        'image_num_key_value_heads': 16
    }
}


class CLIPConfig(PretrainedConfig):
    model_type = "OpenAI_CLIP"
    
    def __init__(
        self,
        image_patch_size = 14,
        image_pos_patch_size = 14,
        image_emb_dim = 1024,
        image_num_heads = 16,
        image_num_key_value_heads = 16,
        image_num_layers = 24,
        image_head_dim = 64,
        image_mlp_dim = 4096,
        image_mlp_activations = ('gelu',),
        image_dropout_rate = 0.0,
        image_default_input_size = (336, 336),
        image_num_pos = 577,
        image_dtype = 'bfloat16',
        image_norm_eps = 1e-5,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        scan_mlp=False,
        scan_attention=False,
        scan_query_chunk_size=1024,
        scan_key_chunk_size=1024,
        scan_mlp_chunk_size=1024,
        initializer_range=0.2,
        remat_block='',
        **kwargs,
    ):
        self.image_patch_size = image_patch_size
        self.image_pos_patch_size = image_pos_patch_size
        self.image_emb_dim = image_emb_dim
        self.image_num_heads = image_num_heads
        self.image_num_layers = image_num_layers
        self.image_head_dim = image_head_dim
        self.image_mlp_dim = image_mlp_dim
        self.image_mlp_activations = image_mlp_activations
        self.image_dropout_rate = image_dropout_rate
        self.image_default_input_size = image_default_input_size
        self.image_num_pos = image_num_pos
        self.image_dtype = image_dtype
        self.image_norm_eps = image_norm_eps
        self.attn_pdrop = attn_pdrop
        self.scan_mlp = scan_mlp
        self.scan_query_chunk_size = scan_query_chunk_size
        self.scan_key_chunk_size = scan_key_chunk_size
        self.scan_mlp_chunk_size = scan_mlp_chunk_size
        self.remat_block = remat_block
        self.image_num_key_value_heads = image_num_key_value_heads
        self.initializer_range = initializer_range
        self.resid_pdrop = resid_pdrop
        self.scan_attention = scan_attention
        
        super().__init__(
            **kwargs,
        )
        
    @classmethod
    def get_default_config(cls, updates=None):
        config = function_args_to_config(cls.__init__)
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

def QuickGELU(x): return x * nn.sigmoid(1.702 * x)
    
class MLP(nn.Module):
    config: CLIPConfig
    dtype: DType = jnp.float32
    param_dtype: DType = jnp.float32
    
    @nn.compact
    def __call__(self, x, deterministic=True):
        cfg = self.config
        x = nn.Dense(
            cfg.image_mlp_dim,
            dtype=self.dtype,
            use_bias=True,
            name='w1',
        )(x)
        x = QuickGELU(x)
        x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))
        x = nn.Dense(
            cfg.image_emb_dim,
            dtype=self.dtype,
            use_bias=True,
            name='w2',
        )(x)

        return x
        

class MultiHeadDotProductAttention(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype=jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self):
        config = self.config
        self.embed_dim = config.image_emb_dim
        self.num_heads = config.image_num_heads
        self.head_dim = config.image_head_dim
        self.num_key_value_heads = config.image_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.wq = nn.Dense(
            config.image_num_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.wk = nn.Dense(
            config.image_num_key_value_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.wv = nn.Dense(
            config.image_num_key_value_heads*self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.wo = nn.Dense(
            config.image_emb_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

    def _split_heads(self, hidden_states, num_heads):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        inputs_q,
        inputs_kv: Optional[Array] = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):

        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q
    
        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_k), self.wv(inputs_v)

        xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
        xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
        xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)
        
        dropout_rng = None
        if not deterministic and self.config.attn_pdrop > 0.0:
            dropout_rng = self.make_rng("dropout")

        # transform boolean mask into float mask
        attn_weights = dot_product_attention_weights(
            xq,
            xk,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
        )

        attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "mp", None, None))
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class ResidualAttentionBlock(nn.Module):
    config: CLIPConfig
    dtype: DType = jnp.float32
    param_dtype: DType = jnp.float32

    def setup(self) -> None:
        attention_module = MultiHeadDotProductAttention
        mlp_module = MLP
        norm_module = nn.LayerNorm
        
        self.attention = attention_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.feed_forward = mlp_module(
            self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.attention_norm = norm_module(
            epsilon=self.config.image_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.ffn_norm = norm_module(
            epsilon=self.config.image_norm_eps,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        
    def __call__(self,
                hidden_states,
                deterministic: bool = True,
                init_cache: bool = False,
                output_attentions: bool = False,
                ):
        
        attn_outputs = self.attention(
            self.attention_norm(hidden_states),
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]

        hidden_states = hidden_states + attn_output

        feed_forward_input = self.ffn_norm(hidden_states)
        feed_forward_hidden_states = self.feed_forward(
            feed_forward_input,
            deterministic,
        )
        feed_forward_hidden_states = with_sharding_constraint(feed_forward_hidden_states, PS(("dp", "fsdp"), None, "mp"))
        hidden_states = hidden_states + feed_forward_hidden_states

        return (hidden_states,) + attn_outputs[1:]



class BlockCollection(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype=jnp.float32

    def setup(self):
        block = ResidualAttentionBlock
        if self.config.remat_block != '':
            block = remat(
                ResidualAttentionBlock, static_argnums=(3, 4, 5),
                policy=get_gradient_checkpoint_policy(self.config.remat_block)
            )
        self.blocks = [
            block(
                self.config,
                name=str(i),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            ) for i in range(self.config.image_num_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states,
                deterministic,
                init_cache,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class VisionTransformer(nn.Module):
    config: CLIPConfig
    dtype: DType = jnp.float32
    param_dtype: DType = jnp.float32
    
    def setup(self):
        cfg = self.config
        scale = cfg.image_emb_dim ** -0.5
        
        self.class_embedding = self.param(
                'class_embedding',
                nn.initializers.normal(stddev=scale), 
                (cfg.image_emb_dim, ),
                self.param_dtype)

        self.positional_embedding = self.param(
                'positional_embedding',
                 nn.initializers.normal(stddev=scale), 
                (cfg.image_num_pos, cfg.image_emb_dim),
                self.param_dtype)

        self.h = BlockCollection(self.config, dtype=self.dtype, param_dtype=self.param_dtype)
        
        self.patch_embedding = nn.Dense(
                features=cfg.image_emb_dim,
                use_bias=False,
                kernel_init=nn.initializers.glorot_uniform(),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name='patch_embedding')
        
        self.pre_ln = nn.LayerNorm(
                epsilon=self.config.image_norm_eps,
                bias_init=nn.initializers.zeros, 
                scale_init= nn.initializers.ones ,
                dtype=jnp.float32, 
                name='pre_ln')

    def add_pos_emb(self, x, patch_num):
        cls_emb = self.positional_embedding[0:1]
        pos_emb = self.positional_embedding[1:]

        pos_emb = jnp.reshape(pos_emb, 
                (int(math.sqrt(pos_emb.shape[0])), int(math.sqrt(pos_emb.shape[0])), pos_emb.shape[1]))
        
        (patch_num_0, patch_num_1) = patch_num
        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            pos_emb = jax.image.resize(pos_emb, (patch_num_0, patch_num_1, pos_emb.shape[-1]), "bicubic")

        pos_emb = jnp.reshape(pos_emb, [-1, pos_emb.shape[-1]])
        x = x + jnp.concatenate([cls_emb[None,:,:], pos_emb[None,:,:]], axis=1)
        return x

    @nn.compact
    def __call__(self,
                x,
                deterministic: bool = True,
                patch_num: Any = (24, 24),
                ):
                
        B, T, N, D = x.shape
        x = jnp.reshape(x, [B*T, N, D])
        mask = jnp.all(x != -1, axis=(1, 2), keepdims=True)

        x = self.patch_embedding(x)
        x = jnp.concatenate([
                jnp.repeat(self.class_embedding[None, None, :], B*T, axis=0),
                x], axis=1)
        
        x = self.add_pos_emb(x, patch_num)
        x = self.pre_ln(x)
        
        x, _, _ = self.h(x, deterministic=deterministic)
        
        x = x[:,1:,:]
        x = x * mask
        x = jnp.reshape(x, [B, T, N, -1])
        
        return x


if __name__ == '__main__':
    config = CLIPConfig()
    model = VisionTransformer(config)
    
    x = jnp.zeros([2,8,576,588])
    params = model.init(jax.random.key(0), x) # Initialization call