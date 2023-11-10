#!/bin/bash

# Block 인덱스 값의 배열 정의
BLOCK_IDX=(1 2 3 4 5 6)
# Head 인덱스 값의 배열 정의
HEAD_IDX=(1 2 3 4 5)

# Block 인덱스에 대해 반복
for current_block in "${BLOCK_IDX[@]}"
do
    # Head 인덱스에 대해 반복
    for current_head in "${HEAD_IDX[@]}"
    do
        echo "Reducing with block = $current_block and head = $current_head"
        sscd/disc_eval_head_ab.py --disc_path /hdd/wi/dataset/DISC2021_mini/ --gpus=2 \
        --output_path=./ \
        --size=224 --preserve_aspect_ratio=true \
        --workers=0 \
        --block_idx=$current_block --head_idx=$current_head \
        --backbone=OFFL_VIT_TINY --dims=192 --model_state=/hdd/wi/sscd-copy-detection/ckpt/exp_ortho_3/allortho[model_3]_OFFL_VIT_TINY_last_True_5.0_cls+pat_192_1107_060517/l1q334sn/checkpoints/epoch=3-step=623.ckpt
    done
done
