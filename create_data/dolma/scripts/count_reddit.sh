export NUM_FOLDS=78
mkdir -p reddit_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/count_data_dolma.py -data_dir data/reddit -fold {1} > reddit_logs/countlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))