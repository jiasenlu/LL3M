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
    "llama2_7b": {
        "vocab_size": 32000,
        "additional_vocab_size": 128,
        "dim": 4096,
        "intermediate_size": 11008,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-5,
        "num_key_value_heads": 32,
        "rope_theta": 10000.0,
    },
    "mistral_7b": {
        "vocab_size": 32000,
        "dim": 4096,
        "intermediate_size": 14336,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-5,
        "max_position_embeddings": 32768,
        "num_key_value_heads": 8,
        "rope_theta": 10000.0, 
    },
    "open_llama_3b": {
        "vocab_size": 32000,
        "dim": 3200,
        "intermediate_size": 8640,
        "n_layers": 26,
        "n_heads": 32,
        "norm_eps": 1e-6,
        "max_position_embeddings": 2048,
        "num_key_value_heads": 32,
        "rope_theta": 10000.0, 
    },
    "tiny_llama_1b": {
        "vocab_size": 32000,
        "dim": 2048,
        "intermediate_size": 5632,
        "n_layers": 22,
        "n_heads": 32,
        "norm_eps": 1e-5,
        "max_position_embeddings": 2048,
        "num_key_value_heads": 4,
        "rope_theta": 10000.0, 
    },
    "gemma_2b": {
        "vocab_size": 256000,
        "dim": 2048,
        "intermediate_size": 16384,
        "n_layers": 18,
        "n_heads": 8,
        "norm_eps": 1e-6,
        "max_position_embeddings": 8192,
        "num_key_value_heads": 1,
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
    params = OPENLLM_STANDARD_CONFIGS[args.model]

    
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
            "wte": {"embedding": jnp.array(ckpt["embed_tokens.weight"].float().numpy(), dtype=jnp.bfloat16)},
            "ln_f": {"kernel": jnp.array(ckpt["norm.weight"].float().numpy(), dtype=jnp.bfloat16)},
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        "wq": {
                            "kernel": jnp.array(inverse_permute(
                                params,
                                ckpt[f"layers.{layer}.self_attn.q_proj.weight"].float().numpy(),
                            ).transpose(), dtype=jnp.bfloat16)
                        },
                        "wk": {
                            "kernel": jnp.array(inverse_permute_kv(
                                params,
                                ckpt[f"layers.{layer}.self_attn.k_proj.weight"].float().numpy(),
                            ).transpose(), dtype=jnp.bfloat16)
                        },
                        "wv": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.self_attn.v_proj.weight"]
                            .float().numpy()
                            .transpose(), dtype=jnp.bfloat16)
                        },
                        "wo": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.self_attn.o_proj.weight"]
                            .float().numpy()
                            .transpose(), dtype=jnp.bfloat16)
                        },
                    },
                    "feed_forward":{
                        "w1": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.mlp.gate_proj.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=jnp.bfloat16)
                        },
                        "w2": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.mlp.down_proj.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=jnp.bfloat16)
                        },
                        "w3": {
                            "kernel": jnp.array(ckpt[f"layers.{layer}.mlp.up_proj.weight"]
                            .float()
                            .numpy()
                            .transpose(), dtype=jnp.bfloat16)
                        },
                    },
                    "attention_norm": {
                        "kernel": jnp.array(ckpt[f"layers.{layer}.input_layernorm.weight"].float().numpy(), dtype=jnp.bfloat16)
                    },
                    "ffn_norm": {
                        "kernel": jnp.array(ckpt[
                            f"layers.{layer}.post_attention_layernorm.weight"
                        ].float().numpy(), dtype=jnp.bfloat16)
                    },
                }
                for layer in tqdm(range(params["n_layers"]))
            },
        },
    }
    
    if 'gemma' not in args.model:
        jax_weights["lm_head"] = {"kernel": jnp.array(ckpt["lm_head.weight"].float().numpy().transpose(), dtype=jnp.bfloat16)}
        
    print(f"Convert weight to ll3m format finished...")
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
    parser = argparse.ArgumentParser(description="hf to ll3m format script")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Need to be converted model weight dir. it is a dir",
    )
    parser.add_argument(
        "--output_file", type=str, help="Save model weight file path, it is a file."
    )
    parser.add_argument(
        "--model",
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
    print(f"model: {args.model}")
    print(f"streaming: {args.streaming}")

    main(args)
