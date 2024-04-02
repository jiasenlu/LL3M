export NUM_FOLDS=26
mkdir -p pes2o_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/count_data_dolma.py -data_dir data/pes2o -fold {1} > pes2o_logs/countlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))