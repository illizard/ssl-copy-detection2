import torch
import timm
import numpy as np
import pdb

def pca_via_svd(W, k):
    U, S, V = torch.svd(W)
    # Ensure k is not greater than the number of singular values
    k = min(k, S.size(0))
    # Perform matrix multiplication using @ operator
    W_tilde = U[:, :k] @ torch.diag(S[:k])
    V = V.view(V.size(0), -1)  # Reshape V to a 2D view
    W_tilde = W_tilde @ V.t()[:k, :]
    return W_tilde


def compute_redundancy(W1, W2, k):
    W_tilde_1 = pca_via_svd(W1, k)
    W_tilde_2 = pca_via_svd(W2, k)
    reconstruction_error_1 = torch.norm(W1 - W_tilde_1, p='fro') ** 2
    reconstruction_error_2 = torch.norm(W2 - W_tilde_2, p='fro') ** 2
    return reconstruction_error_1.item() + reconstruction_error_2.item()


def compute_weight_redun(model):
    redundancy_dict = {}
    for name, param in model.named_parameters():
        if 'qkv.weight' in name:
            # 쿼리, 키, 밸류의 가중치 추출
            qkv_weights = param.view(param.size(0), -1, 3, param.size(1) // 3).transpose(1, 2)
            num_heads = qkv_weights.size(0)
            head_dim = qkv_weights.size(-1)
            # SVD를 위해 head_dim이 1보다 큰지 확인
            if head_dim <= 1:
                continue
            for i in range(num_heads):
                for j in range(i+1, num_heads):
                    for part_index, part_name in enumerate(['query', 'key', 'value']):
                        W1 = qkv_weights[i, part_index].unsqueeze(0)  # 필요한 경우 차원 추가
                        W2 = qkv_weights[j, part_index].unsqueeze(0)  # 필요한 경우 차원 추가
                        # W1과 W2가 2차원인지 확인
                        if W1.ndim < 2:
                            W1 = W1.unsqueeze(0)
                        if W2.ndim < 2:
                            W2 = W2.unsqueeze(0)
                        redundancy = compute_redundancy(W1, W2, head_dim)
                        redundancy_dict[f'head_{i}_vs_head_{j}_{part_name}'] = redundancy
    return redundancy_dict

# Load the model
model_name = "vit_tiny_patch16_224"
model = timm.create_model(model_name, pretrained=True, num_classes=0)
model.eval()  # Set the model to evaluation mode

# Compute the weight redundancy
redundancy_dict = compute_weight_redun(model)

# Print the results
for name, redun in redundancy_dict.items():
    print(f'Redundancy between {name}: {redun}')



