export NUM_FOLDS=1024
mkdir -p c4_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python prep_data_dolma.py -data_dir data/c4 -base_fn gs://unified-io-2-us-east/data -folder_name c4 -file_prefix c4 -fold {1} -num_folds ${NUM_FOLDS} > c4_logs/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))