#!/bin/bash

echo "Start to train the model...."

name="zrrjoint_base_5e-4_100"

dataroot="/home/lab611/Data/PBVS/track1/challengedataset"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

# You can set "--model zrrganjoint" to train LiteISPGAN.

#python train.py \
#    --dataset_name zrr        --model zrrjoint    --name $name         --gcm_coord True  \
#    --pre_ispnet_coord False  --niter 100         --lr_decay_iters 40   --save_imgs False \
#    --batch_size 8         --print_freq 300    --calc_metrics True  --lr 5e-4   -j 8 --writer_name "zrrjoint_base_5e-4_100_b8"\
#    --dataroot $dataroot | tee $LOG


python train.py \
    --dataset_name zrr        --model zrrjoint    --name $name         --gcm_coord True  \
    --pre_ispnet_coord False  --niter 100         --lr_decay_iters 40   --save_imgs False \
    --batch_size 8         --print_freq 300    --calc_metrics True  --lr 5e-4   -j 8 --writer_name "zrrjoint_base_5e-4"\
    --dataroot $dataroot | tee $LOG