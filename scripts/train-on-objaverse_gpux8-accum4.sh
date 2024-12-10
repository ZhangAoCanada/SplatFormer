ngpus=4
accumulate_step=2
batch_size=$((ngpus * accumulate_step))

/home/ZHANGAo_2024/.conda/envs/splatformer/bin/torchrun --nnodes=1 --nproc_per_node=${ngpus} --rdzv-endpoint=localhost:29518 \
    train.py \
    --output_dir=outputs/objaverse_splatformer \
    --gin_file=configs/dataset/objaverse.gin \
    --gin_file=configs/model/ptv3.gin \
    --gin_file=configs/train/default.gin \
    --gin_param="build_trainloader.batch_size="${batch_size} \
    --gin_param="build_trainloader.accumulate_step="${accumulate_step}

