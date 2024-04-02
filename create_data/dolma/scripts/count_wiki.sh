export NUM_FOLDS=2
mkdir -p wiki_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/count_data_dolma.py -data_dir data/wiki -fold {1} > wiki_logs/countlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))