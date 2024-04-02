export NUM_FOLDS=2048
mkdir -p stack_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/prep_data_dolma.py -data_dir data/stack -base_fn gs://unified-io-2-us-east/dolma -folder_name stack -file_prefix stack -fold {1} -num_folds ${NUM_FOLDS} > stack_logs/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))