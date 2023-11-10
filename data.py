import os
import random
import shutil

src_dir = "/hdd/wi/dataset/DISC2021/references/images/references/"
tgt_dir = "/hdd/wi/dataset/DISC2021_exp/references/images/references"
output_dir = "/hdd/wi/dataset/DISC2021_exp/references/images/references_distractor/"  # 이 디렉토리에 선택된 이미지들이 복사됩니다.
num_samples = 10000 # 10k

# 두 디렉토리에서 이미지 이름 가져오기
src_images = set(os.listdir(src_dir))
tgt_images = set(os.listdir(tgt_dir))

# src_dir에만 있는 이미지 선택
unique_images = list(src_images - tgt_images)

# 랜덤하게 이미지 선택
selected_images = random.sample(unique_images, num_samples)

# 선택된 이미지를 output 디렉토리로 복사
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for image in selected_images:
    shutil.copy(os.path.join(src_dir, image), os.path.join(output_dir, image))
