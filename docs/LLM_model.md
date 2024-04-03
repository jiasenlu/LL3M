## Training or finetuning OpenLLM Model from scratch or existing checkpoints.

We first show a simple way to keep training the LLaMA 2 7B model with wikipediate knowledge. 
Download LLaMA checkpoints. 
``` shell
pip install huggingface_hub

python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf --access_token [YOUR_ACCESS_TOKEN]
```

Convert the Official LLaMA Checkpoint to ll3m Format

``` shell
python -m models.openLLM.convert_hf_ll3m \
    --checkpoint_dir='checkpoints/meta-llama/Llama-2-7b-hf' \
    --output_file='checkpoints/meta-llama/Llama-2-7b-ll3m' \
    --model='llama2_7b' \
```
If you use TPU, and google Cloud, you need to upload the checkpoints to google cloud for the TPU cluster access. 

```shell
gsutil -m cp -r checkpoints/meta-llama/Llama-2-7b-ll3m YOUR_GOOGLE_CLOUD_BUCKET_ADDRESS
```

Prepare the data based on the dataset.md, you can use different datalaoder, for TPU, we recommend to use 
the seqio dataloader since it can eazily combine different datasets. Since there is no open checkpoint release
for LLaMA keep training on Wikipedia, Wikibooks data. We will train a checkpoint for demo purpose. 

Next, let start to train the model. 
```shell

```


