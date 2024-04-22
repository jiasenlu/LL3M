"""
Usage:
python convert_hf_to_easylm.py  \
       --checkpoint_dir     /path/hf_format_dir/    \
       --output_file /path/easylm_format.stream   \
       --model_size 7b \
       --streaming
"""
import time
from pathlib import Path
import argparse

import collections
import mlxu
import torch
import flax
import jax.numpy as jnp
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file
import flax.linen as nn
import jax
from data.data_utils import get_default_vocabulary, get_special_token_ids
from module.checkpoint import StreamingCheckpointer

OPENLLM_STANDARD_CONFIGS = {
    "llava-v1.6-vicuna-7b": {
        "vocab_size": 32016,
        "dim": 4096,
        "intermediate_size": 14336,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-5,
        "max_position_embeddings": 4096,
        "num_key_value_heads": 32,
        "rope_theta": 10000.0, 
        "mm_vision_n_layer": 23,
        "mm_vision_tower": "openai/clip-vit-large-patch14-336",
        "mm_hidden_size": 1024,  
    },
    "llava-v1.5-vicuna-7b": {
        "vocab_size": 32000,
        "dim": 4096,
        "intermediate_size": 14336,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-5,
        "max_position_embeddings": 4096,
        "num_key_value_heads": 32,
        "rope_theta": 10000.0, 
        "mm_vision_n_layer": 23,
        "mm_vision_tower": "openai/clip-vit-large-patch14-336",
        "mm_hidden_size": 1024,  
    },
}


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
}

def inverse_permute(params, w):
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    dim = params["dim"]
    reshaped_w = w.reshape(n_heads, 2, dim // n_heads // 2, dim)
    transposed_w = reshaped_w.transpose(0, 2, 1, 3)
    inverted_w = transposed_w.reshape(dim, dim)
    return inverted_w

def inverse_permute_kv(params, w):
    n_layers = params["n_layers"]
    n_kv_heads = params["num_key_value_heads"]
    n_heads = params["n_heads"]
    dim = params["dim"]
    reshaped_w = w.reshape(n_kv_heads, 2, dim // n_heads // 2, dim)
    transposed_w = reshaped_w.transpose(0, 2, 1, 3)
    inverted_w = transposed_w.reshape(n_kv_heads * (dim // n_heads), dim)
    return inverted_w

def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.
  This function is useful to analyze checkpoints that are without need to access
  the exact source code of the experiment. In particular, it can be used to
  extract an reuse various subtrees of the scheckpoint, e.g. subtree of
  parameters.
  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.
  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if '.' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('.', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree

def main(args):
    start = time.time()
    params = OPENLLM_STANDARD_CONFIGS[args.model_size]

    
    if 'v1.5' in args.model_size:
        ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.bin"))
        ckpt = {}
        
        for i in tqdm(range(len(ckpt_paths))):
            checkpoint = torch.load(ckpt_paths[i], map_location="cpu")
            for k, v in checkpoint.items():
                if k.startswith("model."):
                    k = k[6:]
                ckpt[k] = v

        ckpt_paths = sorted(Path(args.vit_checkpoint_dir).glob("*.bin"))
        vit_ckpt = {}

        for i in tqdm(range(len(ckpt_paths))):
            checkpoint = torch.load(ckpt_paths[i], map_location="cpu")
            for k, v in checkpoint.items():
                if k.startswith("vision_model."):
                    k = k[13:]
                    vit_ckpt[k] = v
                    
    else:
        ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.bin"))
        ckpt = {}
        vit_ckpt = {}
        
        for i in tqdm(range(len(ckpt_paths))):
            with safe_open(ckpt_paths[i], framework="pt", device="cpu") as f:
                for key in f.keys():
                    vit_params = False
                    k = key
                    if key.startswith("model."):
                        k = key[6:]
                    if k.startswith("vision_tower.vision_tower.vision_model."):
                        k = k[39:]
                        vit_params = True

                    if vit_params:
                        vit_ckpt[k] = f.get_tensor(key)      
                    else:
                        ckpt[k] = f.get_tensor(key)        

    Dtype = jnp.bfloat16

    # we use image sep token as new line.
    # default_embed_init = nn.initializers.variance_scaling(
    #                 1.0, 'fan_in', 'normal', out_axis=0)
    # new_embedding = jnp.zeros((params['additional_vocab_size'], params["dim"]), Dtype)
    # voc = get_default_vocabulary()
    # special_token_ids = get_special_token_ids()
    
    # col_token_ids = special_token_ids['<im_col>']
    # new_embedding = new_embedding.at[col_token_ids-params['vocab_size']].set(jnp.array(ckpt['image_newline'].float().numpy(), dtype=Dtype))
    # new_embedding = jnp.concatenate([jnp.array(ckpt["embed_tokens.weight"].float().numpy(), dtype=Dtype), new_embedding], axis=0)
    
    print(f"Start convert weight to ll3m format...")
    jax_weights = {
        "transformer": {
            "image_vit": {
                "class_embedding": jnp.array(vit_ckpt['embeddings.class_embedding'].float().numpy(), dtype=Dtype),
                "positional_embedding": jnp.array(vit_ckpt['embeddings.position_embedding.weight'].float().numpy(), dtype=Dtype),
                "patch_embedding": {
                    "kernel": jnp.array(vit_ckpt[f"embeddings.patch_embedding.weight"]
                        .float().numpy()
                        .transpose(2,3,1,0).reshape(-1, params['mm_hidden_size']), dtype=Dtype),
                },
                "pre_ln":{
                    "scale": jnp.array(vit_ckpt[f"pre_layrnorm.weight"].float().numpy(), dtype=Dtype),
                    "bias": jnp.array(vit_ckpt[f"pre_layrnorm.bias"].float().numpy(), dtype=Dtype),                    
                },
                "h":{
                    "%d"
                    % (layer): {
                        "attention": {
                            "wq": {
                                "kernel": jnp.array(
                                    vit_ckpt[f"encoder.layers.{layer}.self_attn.q_proj.weight"]
                                    .float().numpy()
                                    .transpose(),
                                    dtype=Dtype),
                            "bias": jnp.array(vit_ckpt[f"encoder.layers.{layer}.self_attn.q_proj.bias"].float().numpy(), dtype=Dtype),
                            },
                            "wk": {
                                "kernel": jnp.array(
                                    vit_ckpt[f"encoder.layers.{layer}.self_attn.k_proj.weight"]
                                    .float().numpy()
                                    .transpose(),
                                    dtype=Dtype),
                                "bias": jnp.array(vit_ckpt[f"encoder.layers.{layer}.self_attn.k_proj.bias"].float().numpy(), dtype=jnp.float32)
                            },
                            "wv": {
                                "kernel": jnp.array(vit_ckpt[f"encoder.layers.{layer}.self_attn.v_proj.weight"]
                                .float().numpy()
                                .transpose(), dtype=Dtype),
                                "bias": jnp.array(vit_ckpt[f"encoder.layers.{layer}.self_attn.v_proj.bias"].float().numpy(), dtype=jnp.float32)
                            },
                            "wo": {
                                "kernel": jnp.array(vit_ckpt[f"encoder.layers.{layer}.self_attn.out_proj.weight"]
                                .float().numpy()
                                .transpose(), dtype=Dtype),
                                "bias": jnp.array(vit_ckpt[f"encoder.layers.{layer}.self_attn.out_proj.bias"].float().numpy(), dtype=jnp.float32)
                            },
                        },
                        "feed_forward":{
                            "w1": {
                                "kernel": jnp.array(vit_ckpt[f"encoder.layers.{layer}.mlp.fc1.weight"]
                                .float()
                                .numpy()
                                .transpose(), dtype=Dtype),
                                "bias": jnp.array(vit_ckpt[f"encoder.layers.{layer}.mlp.fc1.bias"].float().numpy(), dtype=jnp.float32)
                            },
                            "w2": {
                                "kernel": jnp.array(vit_ckpt[f"encoder.layers.{layer}.mlp.fc2.weight"]
                                .float()
                                .numpy()
                                .transpose(), dtype=Dtype),
                                "bias": jnp.array(vit_ckpt[f"encoder.layers.{layer}.mlp.fc2.bias"].float().numpy(), dtype=jnp.float32)
                            },
                        },
                        "attention_norm": {
                            "scale": jnp.array(vit_ckpt[f"encoder.layers.{layer}.layer_norm1.weight"].float().numpy(), dtype=Dtype),
                            "bias": jnp.array(vit_ckpt[f"encoder.layers.{layer}.layer_norm1.bias"].float().numpy(), dtype=Dtype),                    

                        },
                        "ffn_norm": {
                            "scale": jnp.array(vit_ckpt[f"encoder.layers.{layer}.layer_norm2.weight"].float().numpy(), dtype=Dtype),
                            "bias": jnp.array(vit_ckpt[f"encoder.layers.{layer}.layer_norm2.bias"].float().numpy(), dtype=Dtype),                    
                        },
                    }
                    for layer in tqdm(range(params["mm_vision_n_layer"]))
                }
            },  
            "mm_projector": {
                        "w1": {
                            "kernel": jnp.array(ckpt[f"mm_projector.0.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=Dtype),
                            "bias": jnp.array(ckpt[f"mm_projector.0.bias"].float().numpy(), dtype=Dtype)
                        
                        },
                        "w2": {
                            "kernel": jnp.array(ckpt[f"mm_projector.2.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=Dtype),
                            "bias": jnp.array(ckpt[f"mm_projector.2.bias"].float().numpy(), dtype=Dtype)
                        },
            },
            "wte": {
                "embedding": jnp.array(ckpt["embed_tokens.weight"].float().numpy(), dtype=Dtype),
            },
            "ln_f": {"kernel": jnp.array(ckpt["norm.weight"].float().numpy(), dtype=Dtype)},
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        "wq": {
                            "kernel": jnp.array(inverse_permute(
                                params,
                                ckpt[f"layers.{layer}.self_attn.q_proj.weight"].float().numpy(),
                            ).transpose(), dtype=Dtype)
                        },
                        "wk": {
                            "kernel": jnp.array(inverse_permute_kv(
                                params,
                                ckpt[f"layers.{layer}.self_attn.k_proj.weight"].float().numpy(),
                            ).transpose(), dtype=Dtype)
                        },
                        "wv": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.self_attn.v_proj.weight"]
                            .float().numpy()
                            .transpose(), dtype=Dtype)
                        },
                        "wo": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.self_attn.o_proj.weight"]
                            .float().numpy()
                            .transpose(), dtype=Dtype)
                        },
                    },
                    "feed_forward":{
                        "w1": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.mlp.gate_proj.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=Dtype)
                        },
                        "w2": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.mlp.down_proj.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=Dtype)
                        },
                        "w3": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.mlp.up_proj.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=Dtype)
                        },
                    },
                    "attention_norm": {
                        "kernel": jnp.array(ckpt[f"layers.{layer}.input_layernorm.weight"].float().numpy(), dtype=Dtype)
                    },
                    "ffn_norm": {
                        "kernel": jnp.array(ckpt[
                            f"layers.{layer}.post_attention_layernorm.weight"
                        ].float().numpy(), dtype=Dtype)
                    },
                }
                for layer in tqdm(range(params["n_layers"]))
            },
        },
    }
    
    jax_weights["lm_head"] = {"kernel": jnp.array(ckpt["lm_head.weight"].float().numpy().transpose(), dtype=Dtype)}
    
    print(f"Convert weight to easylm format finished...")
    print(f"Start to save...")

    if args.streaming:
        StreamingCheckpointer.save_train_state_to_file(jax_weights, args.output_file)
    else:
        with mlxu.open_file(args.output_file, "wb") as fout:
            fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True))

    print(
        f"Save finished!!! take time: {time.time() - start} save path: {args.output_file}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hf to easylm format script")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Need to be converted model weight dir. it is a dir",
    )
    parser.add_argument(
        "--output_file", type=str, help="Save model weight file path, it is a file."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="llava-v1.6-vicuna-7b",
        help="model size",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="whether is model weight saved stream format",
    )
    parser.add_argument(
        "--checkpoint_type",
        type=str,
        default='pt',
        help="what is the format of the checkpoit.",
    )
    parser.add_argument(
        "--vit_checkpoint_type",
        type=str,
        default='bin',
        help="what is the format of the checkpoit.",
    )
    
    parser.add_argument(
        "--vit_checkpoint_dir",
        type=str,
        help="Need to be converted model weight dir. it is a dir",
    )
    
    args = parser.parse_args()
    print(f"vit_checkpoint_dir: {args.vit_checkpoint_dir}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"output_file: {args.output_file}")
    print(f"model_size: {args.model_size}")
    print(f"streaming: {args.streaming}")

    main(args)
