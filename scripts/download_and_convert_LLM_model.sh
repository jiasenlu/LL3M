### LLaMA
YOUR_ACCESS_TOKEN = Update your access token here.

pip install huggingface_hub

python -m models.openLLM.convert_hf_easylm \
    --checkpoint_dir='checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T' \
    --output_file='checkpoints/openLLM/TinyLlama-1B-intermediate-step-1431k-3T-ll3m' \
    --model_size='tiny_llama_1b' 

### Mistral
python scripts/download.py --repo_id mistralai/Mistral-7B-v0.1 --access_token $YOUR_ACCESS_TOKEN

python -m models.mistral.convert_hf_easylm \
    --checkpoint_dir='checkpoints/mistralai/Mistral-7B-v0.1' \
    --output_file='checkpoints/mistralai/Mistral-7B-v0.1-ll3m' \
    --streaming

### Phi2
python scripts/download.py --repo_id microsoft/phi-2 --access_token $YOUR_ACCESS_TOKEN --include_suffix "*.safetensors*"

python -m models.openLLM.convert_phi2_hf_ll3m \
    --checkpoint_dir='checkpoints/microsoft/phi-2' \
    --output_file='checkpoints/microsoft/phi-2-ll3m' \
    --streaming \
    --checkpoint_type='safetensors' \
    --model_size='phi_2'

### Gemma
python scripts/download.py --repo_id google/gemma-2b --access_token $YOUR_ACCESS_TOKEN --include_suffix "*.safetensors*"

python -m models.openLLM.convert_hf_ll3m \
    --checkpoint_dir='checkpoints/google/gemma-2b' \
    --output_file='checkpoints/google/gemma-2b-ll3m' \
    --streaming \
    --checkpoint_type='safetensors' \
    --model_size='gemma_2b'

