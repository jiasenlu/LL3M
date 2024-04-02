. $HOME/.LL3M/bin/activate

python3 -m models.llama2.train \
    --mesh_dim='1,4,4' \
    --load_llama_config='7b' \
    --load_checkpoint='params::checkpoints/meta-llama/Llama-2-7b-chat-eazylm' \
    --train_dataset.type='seqio' \
    --train_dataset.seqio_dataset.mixture_or_task_name='flan2_flan2021' \
    --train_dataset.seqio_dataset.batch_size=1 \
    --train_dataset.seqio_dataset.task_feature_field="('targets', 'decoder_loss_weights')" \
    --train_dataset.seqio_dataset.task_feature_lengths='(2048, 2048)' \
    --dtype='bf16' \
    --optimizer.accumulate_gradient_steps=1