import torch
import timm
from timm.models.layers import PatchEmbed
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sscd.models.model import Model
from sscd.lib.util import call_using_args, parse_bool
import argparse
from collections import OrderedDict


from sscd.models import mae_vit, dino_vit
def parse_bool(val):
    return val.lower() in ['true', '1', 'yes']

def load_image_as_tensor(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def compute_distance_matrix(patch_size, num_patches, length):
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])
    return distance_matrix


def compute_mean_attention_dist(patch_size, attention_weights):
    attention_weights = attention_weights[..., 1:, 1:]
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    
    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(mean_distances, axis=-1)
    mean_distances = np.mean(mean_distances, axis=-1)
    return mean_distances


def gather_mads(attention_scores, patch_size: int = 16):
    all_mean_distances = {
        f"block_{i}_mean_dist": compute_mean_attention_dist(
            patch_size=patch_size, attention_weights=attention_weight
        )
        for i, attention_weight in enumerate(attention_scores)
    }
    return all_mean_distances


def visualize_mads(args, all_mads, save_path=None):
    num_blocks = len(all_mads)
    plt.figure(figsize=(10, 6))

    for idx in range(num_blocks):
        mean_distance = all_mads[f"block_{idx}"][0]
        x = [idx] * mean_distance.shape[0]
        y = mean_distance
        plt.scatter(x=x, y=y, label=f"Block {idx}")

    plt.xlabel("Block Index")
    plt.ylabel("MAD")
    plt.legend(loc="lower right")  # 오른쪽 아래로 수정
    plt.title(args.model_state, fontsize=14)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main(args):
    image_path = "/hdd/wi/sscd-copy-detection/sample/sample.jpeg"
    image = load_image_as_tensor(image_path)

    # model_name = "vit_base_patch16_224_in21k"
    model = timm.create_model("vit_base_patch16_224.orig_in21k_ft_in1k", pretrained=True)
    # model = timm.create_model(args.backbone, pretrained=False, num_classes=0)

    # model = dino_vit.__dict__['vit_base'](patch_size=16, num_classes=0)
    # state = torch.load(args.model_state, map_location=torch.device('cpu'))
    # new_state_dict = OrderedDict((k.replace("model.backbone.", ""), v) for k, v in state['state_dict'].items())

    
    # model.load_state_dict(new_state_dict, strict=True)
    print(model)
    model.eval()

    all_mads = {}
    for N in range(12):
        # 어텐션 스코어를 추출하기 위한 feature_extractor 생성
        feature_extractor = create_feature_extractor(
            model, return_nodes=[f'blocks.{N}.attn.softmax'],
            tracer_kwargs={'leaf_modules': [PatchEmbed]})
        
        # 더미 이미지를 사용하여 모델 실행
        with torch.no_grad():
            out = feature_extractor(image)
            
        attention_scores = out[f'blocks.{N}.attn.softmax'].numpy()
        mad = compute_mean_attention_dist(16, attention_scores)
        all_mads[f"block_{N}"] = mad

    # save_path = f"./result"+{args.backbone}+".png"
    save_path = f"./result_vit_base_whole.jpeg"
    visualize_mads(args, all_mads, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--disc_path", required=True, type=str)
    parser.add_argument("--backbone", default="vit_base_patch16_224.dino", type=str)
    parser.add_argument("--dims", default=768, type=int)
    parser.add_argument("--model_state") #checkpoint
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--nodes", default=1, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--size", default=224, type=int, help="Image size for inference")
    args = parser.parse_args()

    main(args)