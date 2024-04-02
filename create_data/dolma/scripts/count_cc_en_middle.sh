export NUM_FOLDS=777
mkdir -p cc_en_middle_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/count_data_dolma.py -data_dir data/cc_en_middle -fold {1} > cc_en_middle_logs/countlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))