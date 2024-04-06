## Different LLM Implementation Overview

In our firt milestone release, we provide the function to convert and finetune [llama2](https://huggingface.co/meta-llama/Llama-2-7b-hf), [mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1), [openLLama](https://huggingface.co/openlm-research/open_llama_3b_v2), [tinyLLama](https://github.com/jzhang38/TinyLlama), [gemma](https://huggingface.co/google/gemma-2b-it), [phi-2](https://huggingface.co/microsoft/phi-2).

The following table shows the difference between the different LLM implementations.

| Model| Tokenizer | vocab_size | #Params|  tie_word_embeddings | has_bias | GQV | norm_module |
| ---- | ---- | ---- | ---- | --- | --- | --- | --- |
| llama2 | llama2 | 32000 | 7B (13/70) | False | False | No | RMSNorm |
| openLLama | llama2 | 32000 | 3B | False | False | No | RMSNorm |
| tinyLLama | llama2 | 32000 | 1.3B | False | False | No | RMSNorm |
| mistral | mistral | 32000 | 7B | False | False | Yes | RMSNorm |
| gemma | gemma | 256000 | 2B (7) | True | False | Yes | GemmaRMSNorm |
| phi-2 | phi-2 | 51200 | 2B | False | True | No | LayerNorm |

Gemma RMSNorm is a variant of RMSNorm that instead of intialized with 1, it is initialized with 0, and added 1 after normalization.

Phi-2 is very different from other models, The philosophy is to use better data instead of overlook model archetectures. For examples, it has bias for all the layers, which is different from other models. It also does not have MLP layernorm, and use LayerNorm instead.

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
    --checkpoint_dir='checkpoints/meta-llama/Llama-22-7b-hf' \
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

Following the data mixture recipe from [Branch-Train-MiX](https://arxiv.org/pdf/2403.07816.pdf), we create a mixture with similar sample rate in `data/mixures.py`

```python
MixtureRegistry.add(
    "wikipedia_mixture",
    [
        ("cc_en", 9.09),
        ("wiki", 90.91),
    ],
    default_rate=1.0)
```

Next, let start to train the model using the `wikipedia_mixture` data.
```shell
python3 -m models.openLLM.train \
    --mesh_dim='-1,2,4' \
    --load_model_config='llama_7b' \
    --checkpointer.save_optimizer_state = True \
    --logger.online=True \
    --logger.output_dir='gs://mm-olmo/LL3M/models/wiki-llama-2' \
    --train_dataset.type='seqio' \
    --train_dataset.seqio_dataset.mixture_or_task_name='wikipedia_mixture' \
    --train_dataset.seqio_dataset.batch_size=32 \
    --train_dataset.seqio_dataset.task_feature_field="('targets', )" \
    --train_dataset.seqio_dataset.task_feature_lengths='(2048, )' \
    --train_dataset.seqio_dataset.shuffle=True \
    --dtype='bf16' \
    --param_dtype='bf16' \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=4 \
    --optimizer.adamw_optimizer.lr=0.0001 \
    --optimizer.adamw_optimizer.end_lr=0.00001 \
    --optimizer.adamw_optimizer.lr_warmup_steps=5000 \
    --optimizer.adamw_optimizer.lr_decay_steps=200000 \
    --total_steps=200000 \
    --save_model_freq=25000 \
    --save_milestone_freq=50000 \
    --gin_bindings='data.data_utils.get_default_vocabulary.tokenizer_type="llama", data.data_utils.get_default_vocabulary.has_extra_token=False' \
    --load_checkpoint='params::YOUR_GOOGLE_CLOUD_BUCKET_ADDRESS' \
```
To train 7B model with sequence length of 2048 on TPU v3, we need to set the `mesh = (-1, 2, 4)`. enable `param_dtype=bf16` can make the batch size == 4. Thus speed up the training process. 

To launch on large TPU PODs, run the following commands 
```shell
python3 tpu_run.py --tpu YOUR_TPU_NAME --script examples/llama2_7B.sh
```

The training log can be find in [here](https://api.wandb.ai/links/jiasenl/ycwul2nr).

The checkpoint and evaluation will upload after 200k steps.
