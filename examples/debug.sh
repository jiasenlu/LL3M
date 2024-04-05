. $HOME/.LL3M/bin/activate
export WANDB_API_KEY="3c1668c7442671a04f67b6024727f53fd2233e12"

# python3 -m models.phi2.train \
#     --mesh_dim='1,-1,1' \
#     --load_model_config='phi_2' \
#     --train_dataset.type='seqio' \
#     --train_dataset.seqio_dataset.mixture_or_task_name='dolma_mixture' \
#     --train_dataset.seqio_dataset.task_feature_field="('targets',)" \
#     --train_dataset.seqio_dataset.task_feature_lengths='(4096,)' \
#     --train_dataset.seqio_dataset.pack=True \
#     --train_dataset.seqio_dataset.shuffle=True \
#     --train_dataset.seqio_dataset.batch_size=8 \
#     --dtype='bf16' \
#     --load_checkpoint='params::checkpoints/microsoft/phi-2-ll3m' \
#     --optimizer.type='adamw' \
#     --optimizer.accumulate_gradient_steps=8 \
#     --optimizer.adamw_optimizer.lr=0.00005 \
#     --optimizer.adamw_optimizer.end_lr=0 \
#     --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
#     --optimizer.adamw_optimizer.lr_decay_steps=1000000 \

# python3 -m models.openMoE.train \
#     --mesh_dim='1,1,1,-1' \
#     --load_model_config='debug' \
#     --train_dataset.text_processor.fields='article' \
#     --train_dataset.type='huggingface' \
#     --train_dataset.huggingface_dataset.path='cnn_dailymail' \
#     --train_dataset.huggingface_dataset.name '3.0.0' \
#     --train_dataset.huggingface_dataset.split 'validation' \
#     --train_dataset.huggingface_dataset.batch_size 16 \
#     --dtype='bf16' \
#     --optimizer.type='adamw' \
#     --load_checkpoint='params::gs://unified-io-2/mixtral-tokenizer/Mixtral-8x7B-Instruct-v0.1-eazylm-2layer-1' 


# python3 -m models.mixtral.train \
#     --mesh_dim='1,8,-1,1' \
#     --load_mixtral_config='8x0.5b' \
#     --train_dataset.type='seqio' \
#     --train_dataset.seqio_dataset.mixture_or_task_name='cc_en' \
#     --train_dataset.seqio_dataset.batch_size=8 \
#     --train_dataset.seqio_dataset.task_feature_field="('targets',)" \
#     --train_dataset.seqio_dataset.task_feature_lengths='(2048,)' \
#     --logger.online=True \
#     --logger.project="LL3M" \
#     --dtype='bf16' \
#     --optimizer.type='adamw' \
#     --save_model_freq=100 \
#     --save_milestone_freq=200 \
#     --logger.output_dir='gs://unified-io-2/moe/mixtral8x0.5' \
#     --optimizer.accumulate_gradient_steps=8

# python3 -m models.openMoE.train \
#     --mesh_dim='1,8,1,1' \
#     --log_freq=64 \
#     --load_model_config='debug' \
#     --logger.online=False \
#     --logger.project="LL3M" \
#     --logger.entity="jiasenl" \
#     --logger.output_dir='gs://unified-io-2/moe/mixtral8x7B_from_mistral' \
#     --dtype='bf16' \
#     --train_dataset.type='seqio' \
#     --train_dataset.seqio_dataset.mixture_or_task_name='dolma_mixture' \
#     --train_dataset.seqio_dataset.task_feature_field="('targets',)" \
#     --train_dataset.seqio_dataset.task_feature_lengths='(2048,)' \
#     --train_dataset.seqio_dataset.pack=True \
#     --train_dataset.seqio_dataset.shuffle=True \
#     --train_dataset.seqio_dataset.batch_size=8 \
#     --optimizer.type='adamw' \
#     --optimizer.accumulate_gradient_steps=8 \
#     --optimizer.adamw_optimizer.lr=0.00005 \
#     --optimizer.adamw_optimizer.end_lr=0 \
#     --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
#     --optimizer.adamw_optimizer.lr_decay_steps=1000000 \
#     # --total_steps=1000000 \
#     # --save_model_freq=10000 \
#     # --load_checkpoint='params::gs://unified-io-2-us-east/moe/Mixtral-7B-from-mistral-eazylm-debug'

# python3 -m models.mistral.train \
#     --mesh_dim='1,4,2' \
#     --load_mistral_config='7b' \
#     --train_dataset.type='seqio' \
#     --train_dataset.seqio_dataset.mixture_or_task_name='cc_en' \
#     --train_dataset.seqio_dataset.batch_size=4 \
#     --train_dataset.seqio_dataset.task_feature_field="('targets',)" \
#     --train_dataset.seqio_dataset.task_feature_lengths='(2048,)' \
#     --logger.online=False \
#     --logger.project="LL3M" \
#     --dtype='bf16' \
#     --optimizer.type='adamw' \
#     --save_model_freq=100 \
#     --save_milestone_freq=200 \
#     --logger.output_dir='gs://unified-io-2/moe/debug' \
#     # --optimizer.accumulate_gradient_steps=8
#     # --load_checkpoint='params::gs://unified-io-2/mixtral-tokenizer/Mixtral-8x7B-Instruct-v0.1-eazylm-2layer-1' 


# python3 -m models.aries.train \
#     --mesh_dim='1,-1,1' \
#     --load_model_config='default::debug||debug' \
#     --train_dataset.type='seqio' \
#     --train_dataset.seqio_dataset.mixture_or_task_name='coco_caption_2017' \
#     --train_dataset.seqio_dataset.batch_size=8 \
#     --train_dataset.seqio_dataset.task_feature_field="('targets', 'decoder_loss_weights', 'images', 'image_positions', 'image_input_idx')" \
#     --train_dataset.seqio_dataset.task_feature_lengths='(1024, 1024, 8, 8, 8)' \
#     --dtype='bf16' \
#     --gin_bindings='data.data_utils.get_default_vocabulary.tokenizer_type = "llama", data.data_utils.get_default_vocabulary.has_extra_token = True' \
#     --optimizer.accumulate_gradient_steps=2
    # --load_checkpoint='params::checkpoints/aries/llama-2-7b_vit'


python3 -m models.soupLLM.train \
    --mesh_dim='1,4, 1,2' \
    --load_model_config='llama_7b' \
    --train_dataset.type='seqio' \
    --train_dataset.seqio_dataset.mixture_or_task_name='wikipedia_mixture' \
    --train_dataset.seqio_dataset.batch_size=1 \
    --train_dataset.seqio_dataset.task_feature_field="('targets', )" \
    --train_dataset.seqio_dataset.task_feature_lengths='(2048, )' \
    --train_dataset.seqio_dataset.shuffle=True \
    --dtype='bf16' \
    --optimizer.accumulate_gradient_steps=1 \
    # --load_checkpoint='params::checkpoints/soup_debug-ll3m' \
    # --gin_bindings='data.data_utils.get_default_vocabulary.tokenizer_type="llama", data.data_utils.get_default_vocabulary.has_extra_token=True, data.preprocessors.image_to_patches_and_tokens.image_token_length_h=24, data.preprocessors.image_to_patches_and_tokens.image_token_length_w=24'

# python3 -m models.llama2.train \
#     --mesh_dim='1,1,-1' \
#     --load_llama_config='7b' \
#     --load_checkpoint='params::checkpoints/Llama-2-7b-chat-eazylm-debug' \
#     --train_dataset.text_processor.fields='article' \
#     --train_dataset.type='huggingface' \
#     --train_dataset.huggingface_dataset.path='cnn_dailymail' \
#     --train_dataset.huggingface_dataset.name '3.0.0' \
#     --train_dataset.huggingface_dataset.split 'validation' \
#     --train_dataset.huggingface_dataset.batch_size 1 \
#     --dtype='bf16' 

# python3 -m models.llama2.train \
#     --mesh_dim='-1,2,2' \
#     --load_llama_config='debug' \
#     --train_dataset.type='seqio' \
#     --train_dataset.seqio_dataset.mixture_or_task_name='flan2_flan2021' \
#     --train_dataset.seqio_dataset.batch_size=16 \
#     --train_dataset.seqio_dataset.task_feature_field="('targets', 'decoder_loss_weights', 'images', 'image_positions', 'image_input_idx')" \
#     --train_dataset.seqio_dataset.task_feature_lengths='(2048, 2048, 8, 8, 8)' \
#     --dtype='bf16' \
#     --optimizer.accumulate_gradient_steps=2

# python3 -m models.llama2.train \
#     --mesh_dim='-1,2,2' \
#     --load_llama_config='debug' \
#     --train_dataset.type='seqio' \
#     --train_dataset.seqio_dataset.mixture_or_task_name='flan2_flan2021' \
#     --train_dataset.seqio_dataset.batch_size=128 \
#     --train_dataset.seqio_dataset.task_feature_field="('targets', 'decoder_loss_weights')" \
#     --train_dataset.seqio_dataset.task_feature_lengths='(2048, 2048)' \
#     --dtype='bf16' \
#     --optimizer.accumulate_gradient_steps=2
