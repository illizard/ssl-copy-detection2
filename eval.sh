# sscd/disc_eval.py --disc_path /hdd/wi/dataset/DISC2021_mini/  --gpus=2 \
#   --output_path=./ \
#   --size=288 --preserve_aspect_ratio=true \
#   --backbone=CV_RESNET50 --dims=512 --model_state=./sscd_disc_blur.torchscript.pt

sscd/disc_eval.py --disc_path /hdd/wi/dataset/DISC2021_mini/ --gpus=2 \
  --output_path=./ \
  --size=224 --preserve_aspect_ratio=true \
  --workers=0 \
  --backbone=OFFL_VIT_TINY --dims=192 --checkpoint=/hdd/wi/sscd-copy-detection/ckpt/[model_1]_OFFL_VIT_TINY_last_false_cls_192dims_model_torchscript.pt