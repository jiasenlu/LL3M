export NUM_FOLDS=256
mkdir -p reddit_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/prep_data_dolma.py -data_dir data/reddit -base_fn gs://unified-io-2-us-east/dolma -folder_name reddit -file_prefix reddit -fold {1} -num_folds ${NUM_FOLDS} > reddit_logs/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))