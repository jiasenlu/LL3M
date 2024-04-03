. $HOME/.LL3M/bin/activate

python3 -m models.openLLM.train \
    --mesh_dim='-1,2,4' \
    --load_model_config='llama_7b_flash' \
    --load_checkpoint='params::checkpoints/meta-llama/Llama-2-7b-ll3m' \
    --train_dataset.type='seqio' \
    --train_dataset.seqio_dataset.mixture_or_task_name='wiki' \
    --train_dataset.seqio_dataset.batch_size=1 \
    --train_dataset.seqio_dataset.task_feature_field="('targets', )" \
    --train_dataset.seqio_dataset.task_feature_lengths='(2048, )' \
    --dtype='bf16' \
    --optimizer.accumulate_gradient_steps=1 \
    --gin_bindings='data.data_utils.get_default_vocabulary.tokenizer_type="llama", data.data_utils.get_default_vocabulary.has_extra_token=False'