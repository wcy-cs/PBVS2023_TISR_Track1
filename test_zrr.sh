#!/bin/bash
echo "Start to test the model...."

name="zrrjoint_base_5e-4"
dataroot="/home/lab611/Data/PBVS/track1/challengedataset"

python test.py \
    --model zrrjoint   --name $name      --dataset_name zrr   --pre_ispnet_coord False  --gcm_coord True \
    --load_iter 60     --save_imgs True  --calc_metrics  False --gpu_id 0        --visual_full_imgs False\
    --dataroot $dataroot

python metrics.py  --name $name --dataroot $dataroot
