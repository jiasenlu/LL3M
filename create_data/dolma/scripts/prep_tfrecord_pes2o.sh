export NUM_FOLDS=256
mkdir -p pes2o_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/prep_data_dolma.py -data_dir data/pes2o -base_fn gs://unified-io-2-us-east/dolma -folder_name pes2o -file_prefix pes2o -fold {1} -num_folds ${NUM_FOLDS} > pes2o_logs/trainlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))