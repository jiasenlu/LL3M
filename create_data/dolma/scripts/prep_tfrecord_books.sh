export NUM_FOLDS=128
mkdir -p books_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/prep_data_dolma.py -data_dir data/books -base_fn gs://unified-io-2-us-east/dolma -folder_name books -file_prefix books -fold {1} -num_folds ${NUM_FOLDS} > books_logs/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))