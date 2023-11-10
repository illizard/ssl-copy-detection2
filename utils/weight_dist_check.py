import torch
import timm
import matplotlib.pyplot as plt
import os

def plot_weight_histograms(weights_distribution, save_path):
    # num_heads는 각 attention 유형(query, key, value)에 대한 헤드 수를 가정합니다.
    num_heads = 3
    # weights_distribution의 길이는 서브플롯의 총 수와 일치해야 합니다.
    # 각 가중치 집합(쿼리, 키, 값)에 대한 서브플롯을 생성합니다.
    plt.figure(figsize=(15, 5))
    for i, (key, stats) in enumerate(weights_distribution.items()):
        for j in range(num_heads):
            # 서브플롯을 생성합니다.
            ax = plt.subplot(1, len(weights_distribution), i+1)
            weight_flat = torch.cat([w.flatten() for w in stats[j]['weights']])
            # 가중치 히스토그램을 그립니다.
            ax.hist(weight_flat.detach().cpu().tolist(), bins=50, alpha=0.5, label=f'Head {j}')
            ax.set_title(f'{key} head {j}')
            ax.set_xlabel('Weight values')
            ax.set_ylabel('Frequency')
            ax.legend()
    plt.tight_layout()
    # 지정된 경로에 플롯을 저장합니다.
    plt.savefig(save_path)
    plt.close()  # 메모리를 해제하기 위해 피겨를 닫습니다.




def check_weights_distribution(model):
    stats_dict = {}
    for name, param in model.named_parameters():
        if 'qkv.weight' in name:
            wq_all, wk_all, wv_all = param.chunk(3, dim=0)
            # Assuming the model has 3 heads, this may need to be adjusted for other models
            w_qs = wq_all.chunk(3, dim=0)
            w_ks = wk_all.chunk(3, dim=0)
            w_vs = wv_all.chunk(3, dim=0)
            
            for i, weights in enumerate([w_qs, w_ks, w_vs]):
                key_prefix = ["query", "key", "value"][i]
                stats = []
                for j, w in enumerate(weights):
                    stats.append({
                        "weights": w
                    })
                stats_dict[f'{key_prefix}_{name}'] = stats
    return stats_dict

if __name__ == '__main__':

    # Load the pre-trained ViT model
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)

    # Compute the statistics for the weights distribution
    weights_distribution = check_weights_distribution(model)

    # Specify the path where you want to save the plot
    save_path = '../histograms/sma.png'

    # Plot and save the weights distribution
    plot_weight_histograms(weights_distribution, save_path)
