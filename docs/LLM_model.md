### LLaMA

Download LLaMA checkpoints. 
``` shell
pip install huggingface_hub

python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf --access_token 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
```

Convert the Official LLaMA Checkpoint to eazylm Format

``` shell
python -m models.llama2.convert_hf_easylm \
    --checkpoint_dir='checkpoints/meta-llama/Llama-2-7b-chat-hf' \
    --output_file='checkpoints/meta-llama/Llama-2-7b-chat-eazylm' 
```

``` shell
python -m models.openLLM.convert_hf_easylm \
    --checkpoint_dir='checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' \
    --output_file='checkpoints/openLLM/TinyLlama-1B-intermediate-step-1431k-3T-eazylm' \
    --model_size='tiny_llama_1b' 
```

### Mistral
Download Mistral checkpoints.
```shell
python scripts/download.py --repo_id mistralai/Mistral-7B-v0.1 --access_token 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
```

Convert the official Mistral checkpoint to ll3m format. 
```shell
python -m models.mistral.convert_hf_easylm \
    --checkpoint_dir='checkpoints/mistralai/Mistral-7B-v0.1' \
    --output_file='checkpoints/mistralai/Mistral-7B-v0.1-eazylm' \
    --streaming
```


### Phi2
Download Mistral checkpoints.
```shell
python scripts/download.py --repo_id microsoft/phi-2 --access_token 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 --include_suffix "*.safetensors*"
```

```shell
python -m models.openLLM.convert_phi2_hf_ll3m \
    --checkpoint_dir='checkpoints/microsoft/phi-2' \
    --output_file='checkpoints/microsoft/phi-2-ll3m' \
    --streaming \
    --checkpoint_type='safetensors' \
    --model_size='phi_2'
```


### Gemma

Gemma 2b

Download Mistral checkpoints.
```shell
python scripts/download.py --repo_id google/gemma-2b --access_token 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 --include_suffix "*.safetensors*"
```

```shell
python -m models.openLLM.convert_hf_ll3m \
    --checkpoint_dir='checkpoints/google/gemma-2b' \
    --output_file='checkpoints/google/gemma-2b-ll3m' \
    --streaming \
    --checkpoint_type='safetensors' \
    --model_size='gemma_2b'
```

