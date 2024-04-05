
### current our model should include these model as soup base. 
llama2, vicuna, codellama, Llemma, meditron, llava1.6, llama2-finetune-on-wiki

### Download the checkpoint

```shell
bash scripts/download_moe.sh
```

### Load all paramters and convert to single file for model soup.

```shell
python -m models.soupLLM.convert_hf_ll3m \
    --checkpoint_dir='checkpoints/' \
    --output_file='checkpoints/soupLLM/soup_debug-ll3m' \
    --streaming
```

### 