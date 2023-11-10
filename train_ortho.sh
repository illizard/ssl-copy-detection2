# !/bin/bash
# 1. ep 30
# 2. train 50k / q 10k / ref 20k (gt 10k + dist 10k)
# 3. bs 512

######################################################################################
# orthogonal exp 
######################################################################################
# InfoNCE loss + ent reg + orthogonal ### Vit Tiny Original ##

    # --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/train_10k/images/train \
    # --val_dataset_path=/hdd/wi/dataset/DISC2021_exp \
 
# LAMBS=(1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 0 5 10 50 100)  # Extended list of lambda values
# LAMBS=(1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1e0 0 5 10 50 100 1e-4 5e-4)  # Extended list of lambda values
LAMBS=(10)  # Extended list of lambda values

for current_lamb in "${LAMBS[@]}"
do
    echo "Running with lamb = $current_lamb"

    MASTER_ADDR="localhost" MASTER_PORT="15084" NODE_RANK="0" WORLD_SIZE=2 \
    ./sscd/train_ortho.py --nodes=1 --gpus=2 --batch_size=128 --weight_decay=0.0 \
    --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/images/train_20k/ \
    --val_dataset_path=/hdd/wi/dataset/DISC2021_exp \
    --entropy_weight=30 --epochs=30 --augmentations=ADVANCED --mixup=true  \
    --output_path=./ckpt/exp_test/ \
    --backbone=OFFL_VIT_TINY --dims=384 \
    --orthogonality=True --lamb=$current_lamb \
    --token=cls+pat \
    --blk=last \
    --model_idx=3 \

done