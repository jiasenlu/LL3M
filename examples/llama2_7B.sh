. $HOME/.LL3M/bin/activate
export WANDB_API_KEY=YOUR_WANDB_KEY

python3 -m models.openLLM.train \
    --mesh_dim='-1,4,4' \
    --load_model_config='llama_7b' \
    --checkpointer.save_optimizer_state = True \
    --logger.online=True \
    --logger.project="LL3M" \
    --logger.entity="jiasenl" \
    --logger.output_dir='gs://mm-olmo/LL3M/models/wiki-llama-2' \
    --train_dataset.type='seqio' \
    --train_dataset.seqio_dataset.mixture_or_task_name='wikipedia_mixture' \
    --train_dataset.seqio_dataset.batch_size=32 \
    --train_dataset.seqio_dataset.task_feature_field="('targets', )" \
    --train_dataset.seqio_dataset.task_feature_lengths='(2048, )' \
    --train_dataset.seqio_dataset.shuffle=True \
    --dtype='bf16' \
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
    --load_checkpoint='params::gs://mm-olmo/LL3M/checkpoints/Llama-2-7b-ll3m' \
