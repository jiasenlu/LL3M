export NUM_FOLDS=128
mkdir -p wiki_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/prep_data_dolma.py -data_dir data/wiki -base_fn gs://unified-io-2-us-east/dolma -folder_name wiki -file_prefix wiki -fold {1} -num_folds ${NUM_FOLDS} > wiki_logs/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))