export NUM_FOLDS=149
mkdir -p stack_logs
parallel -j $(nproc --all) --will-cite "CUDA_VISIBLE_DEVICES='' python scripts/count_data_dolma.py -data_dir data/stack -fold {1} > stack_logs/countlog{1}.txt" ::: $(seq 0 $((${NUM_FOLDS}-1)))