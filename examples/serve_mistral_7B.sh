. $HOME/.LL3M/bin/activate

python -m models.mistral.serve \
    --load_mistral_config='7b' \
    --load_checkpoint='params::checkpoints/mistralai/Mistral-7B-v0.1-eazylm' \
    --mesh_dim='1,-1,1' \
    --dtype='fp16' \
    --input_length=1024 \
    --seq_length=2048 \
    --lm_server.batch_size=1 \
    --lm_server.port=35009 \
    --lm_server.pre_compile='chat' \
    --lm_server.chat_user_prefix='[INST]'\
    --lm_server.chat_user_suffix='[/INST]'
    # --lm_server.chat_prepend_text='<<SYS>>\n You are a helpful, respectful and honest assistant. \
    # Always answer as helpfully as possible, while being safe.  \
    # Your answers should not include any harmful, unethical, \
    # racist, sexist, toxic, dangerous, or illegal content. \
    # Please ensure that your responses are socially unbiased \
    # and positive in nature. If a question does not make any sense, \
    # or is not factually coherent, explain why instead of answering \
    # something not correct. If you don't know the answer to a question, \
    # please don't share false information. \n<</SYS>>\n\n' \