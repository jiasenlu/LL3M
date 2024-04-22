### Llava 1.5

Download the checkpoint:
```shell
python scripts/download.py --repo_id liuhaotian/llava-v1.5-7b
```
Download the openai clip checkpoint:
```shell
python scripts/download.py --repo_id openai/clip-vit-large-patch14-336
```


```shell
python -m models.llava.convert_hf_ll3m \
    --checkpoint_dir='checkpoints/liuhaotian/llava-v1.5-7b' \
    --vit_checkpoint_dir='checkpoints/openai/clip-vit-large-patch14-336' \
    --output_file='checkpoints/llava/llava-v1.5-7b-ll3m' \
    --model_size='llava-v1.5-vicuna-7b' \
    --streaming
```