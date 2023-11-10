# !/bin/bash
# '''
# 1. ep 30
# 2. train 50k / q 10k / ref 20k (gt 10k + dist 10k)

# backbone
# desc_size
# token
# blk
# model index
# '''

## Vit Tiny Original ##
### model 1 ### # 그냥 vit tiny 기본 #
# echo "MODEL 1"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=2 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_mini/train_10k/images/train \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_mini \
#   --entropy_weight=30 --epochs=1 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_test/ \
#   --backbone=OFFL_VIT_TINY  --desc_size=192 \
#   --orthogonality=false --lamb=0.1 \
#   --token=cls \
#   --blk=last \
#   --model_idx=1 \

# ### model 2 ### 
# echo "MODEL 2"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=2 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_mini/train_10k/images/train \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_mini \
#   --entropy_weight=30 --epochs=1 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_test/ \
#   --backbone=OFFL_VIT_TINY  --desc_size=192 \
#   --orthogonality=false --lamb=0.1 \
#   --token=pat \
#   --blk=last \
#   --model_idx=2 \

### model 3 ### 
echo "MODEL 3"
MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
  ./sscd/train_og.py --nodes=1 --gpus=2 --batch_size=128 \
  --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/images/train_20k \
  --val_dataset_path=/hdd/wi/dataset/DISC2021_exp \
  --entropy_weight=30 --epochs=30 --augmentations=ADVANCED --mixup=true  \
  --output_path=./ckpt/head_test/ \
  --backbone=OFFL_VIT_TINY


# ### model 4 ### 
# echo "MODEL 4"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=2 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_mini/train_10k/images/train \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_mini \
#   --entropy_weight=30 --epochs=1 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_test/ \
#   --backbone=OFFL_VIT_TINY  --desc_size=192 \
#   --orthogonality=false --lamb=0.1 \
#   --token=cls+pat+lin \
#   --blk=last \
#   --model_idx=4 \

# ### model 5 ### 
# echo "MODEL 5"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=2 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_mini/train_10k/images/train \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_mini \
#   --entropy_weight=30 --epochs=1 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_test/ \
#   --backbone=MY_DTOP_VIT_192  --desc_size=192 \
#   --orthogonality=false --lamb=0.1 \
#   --token=cls \
#   --blk=6blks \
#   --model_idx=5 \

# ### model 6 ### 
# echo "MODEL 6"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=2 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_mini/train_10k/images/train \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_mini \
#   --entropy_weight=30 --epochs=1 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_test/ \
#   --backbone=MY_DTOP_VIT_192  --desc_size=192 \
#   --orthogonality=false --lamb=0.1 \
#   --token=pat \
#   --blk=6blks \
#   --model_idx=6 \

# ### model 7 ### 
# echo "MODEL 7"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=128 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/images/train_10k_old \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_exp \
#   --entropy_weight=30 --epochs=50 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_dtop/ \
#   --backbone=MY_DTOP_VIT_192  --desc_size=384 \
#   --orthogonality=false --lamb=0.1 \
#   --token=cls+pat \
#   --blk=6blks \
#   --model_idx=7 \

# ### model 8 ### 
# echo "MODEL 8"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=2 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_mini/train_10k/images/train \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_mini \
#   --entropy_weight=30 --epochs=1 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_test/ \
#   --backbone=MY_DTOP_VIT_192  --desc_size=192 \
#   --orthogonality=false --lamb=0.1 \
#   --token=cls+pat+lin \
#   --blk=6blks \
#   --model_idx=8 \

# ### model 9 ### 
# echo "MODEL 9"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=2 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_mini/train_10k/images/train \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_mini \
#   --entropy_weight=30 --epochs=1 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_test/ \
#   --backbone=MY_DTOP_VIT_384  --desc_size=384 \
#   --orthogonality=false --lamb=0.1 \
#   --token=cls+pat+lin \
#   --blk=6blks \
#   --model_idx=9 \

# ### model 10 ### 
# echo "MODEL 10"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=128 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/train_10k/images/train \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_exp \
#   --entropy_weight=30 --epochs=50 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_test/ \
#   --backbone=MY_DTOP_VIT_192  --desc_size=384 \
#   --orthogonality=false --lamb=0.1 \
#   --token=cls+pat \
#   --blk=6blks \
#   --model_idx=10 \


###vit gal
### model 1 ### # 그냥 vit tiny 기본 #
# echo "MODEL 1"
# MASTER_ADDR="localhost" MASTER_PORT="15083" NODE_RANK="0" WORLD_SIZE=2 \
#   ./sscd/train_dtop.py --nodes=1 --gpus=2 --batch_size=128 \
#   --train_dataset_path=/hdd/wi/dataset/DISC2021_exp/images/train_20k \
#   --val_dataset_path=/hdd/wi/dataset/DISC2021_exp \
#   --entropy_weight=30 --epochs=50 --augmentations=ADVANCED --mixup=true  \
#   --output_path=./ckpt/exp_ortho_3/ \
#   --backbone=OFFL_VIT_TINY  --desc_size=384 \
#   --orthogonality=false --lamb=0.1 \
#   --token=cls \
#   --blk=last \
#   --model_idx=3 \