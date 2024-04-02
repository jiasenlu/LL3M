export NUM_FOLDS=4096
mkdir -p cc_en_middle_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/prep_data_dolma.py -data_dir data/cc_en_middle -base_fn gs://unified-io-2-us-east/dolma -folder_name cc_en_middle -file_prefix cc_en_middle -fold {1} -num_folds ${NUM_FOLDS} > cc_en_middle_logs/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))