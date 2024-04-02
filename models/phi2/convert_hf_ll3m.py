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

import mlxu
import torch
import flax
import jax.numpy as jnp
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file

from module.checkpoint import StreamingCheckpointer

OPENLLM_STANDARD_CONFIGS = {
    "phi_2": {
        "vocab_size": 51200,
        "dim": 2560,
        "intermediate_size": 10240,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-5,
        "max_position_embeddings": 2048,
        "num_key_value_heads": 32,
        "rope_theta": 10000.0,
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

def main(args):
    start = time.time()
    params = OPENLLM_STANDARD_CONFIGS[args.model_size]

    
    if args.checkpoint_type == 'safetensors':
        ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.safetensors"))
        ckpt = {}
        
        for i in tqdm(range(len(ckpt_paths))):
            with safe_open(ckpt_paths[i], framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("model."):
                        k = key[6:]
                    else:
                        k = key
                    ckpt[k] = f.get_tensor(key)        
    else:
        ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.bin"))
        ckpt = {}

        for i in tqdm(range(len(ckpt_paths))):
            checkpoint = torch.load(ckpt_paths[i], map_location="cpu")
            for k, v in checkpoint.items():
                if k.startswith("model."):
                    k = k[6:]
                ckpt[k] = v
        
    print(f"Start convert weight to ll3m format...")
    jax_weights = {
        "transformer": {
            "wte": {"embedding": jnp.array(ckpt["embed_tokens.weight"].float().numpy(), dtype=jnp.float32)},
            "ln_f": {
                "scale": jnp.array(ckpt["final_layernorm.weight"].float().numpy(), dtype=jnp.float32),
                "bias": jnp.array(ckpt["final_layernorm.bias"].float().numpy(), dtype=jnp.float32)
            },
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        "wq": {
                            "kernel": jnp.array(inverse_permute(
                                params,
                                ckpt[f"layers.{layer}.self_attn.q_proj.weight"].float().numpy(),).transpose(), dtype=jnp.float32),
                            "bias": jnp.array(ckpt[f"layers.{layer}.self_attn.q_proj.bias"], dtype=jnp.float32)
                        },
                        "wk": {
                            "kernel": jnp.array(inverse_permute_kv(
                                params,
                                ckpt[f"layers.{layer}.self_attn.k_proj.weight"].float().numpy(),
                            ).transpose(), dtype=jnp.float32),
                            "bias": jnp.array(ckpt[f"layers.{layer}.self_attn.k_proj.bias"], dtype=jnp.float32)
                        },
                        "wv": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.self_attn.v_proj.weight"]
                            .float().numpy()
                            .transpose(), dtype=jnp.float32),
                            "bias": jnp.array(ckpt[f"layers.{layer}.self_attn.v_proj.bias"], dtype=jnp.float32)
                        },
                        "wo": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.self_attn.dense.weight"]
                            .float().numpy()
                            .transpose(), dtype=jnp.float32),
                            "bias": jnp.array(ckpt[f"layers.{layer}.self_attn.dense.bias"], dtype=jnp.float32)
                        },
                    },
                    "feed_forward":{
                        "w1": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.mlp.fc1.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=jnp.float32),
                            "bias": jnp.array(ckpt[f"layers.{layer}.mlp.fc1.bias"], dtype=jnp.float32)
                        },
                        "w2": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.mlp.fc2.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=jnp.float32),
                            "bias": jnp.array(ckpt[f"layers.{layer}.mlp.fc2.bias"], dtype=jnp.float32)
                        },
                    },
                    "attention_norm": {
                        "scale": jnp.array(ckpt[f"layers.{layer}.input_layernorm.weight"].float().numpy(), dtype=jnp.float32),
                        "bias": jnp.array(ckpt[f"layers.{layer}.input_layernorm.bias"].float().numpy(), dtype=jnp.float32)
                    },
                }
                for layer in tqdm(range(params["n_layers"]))
            },
        },
        "lm_head": {
            "kernel": jnp.array(ckpt["lm_head.weight"].float().numpy().transpose(), dtype=jnp.float32),
            "bias": jnp.array(ckpt["lm_head.bias"].float().numpy(), dtype=jnp.float32)
        },
    }
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
        default="7b",
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

    args = parser.parse_args()

    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"output_file: {args.output_file}")
    print(f"model_size: {args.model_size}")
    print(f"streaming: {args.streaming}")

    main(args)
