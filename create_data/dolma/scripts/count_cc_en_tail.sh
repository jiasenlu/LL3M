export NUM_FOLDS=1493
mkdir -p cc_en_tail_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/count_data_dolma.py -data_dir data/cc_en_tail -fold {1} > cc_en_tail_logs/countlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))