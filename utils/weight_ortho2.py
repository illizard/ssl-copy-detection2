import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timm.models import create_model
from datetime import datetime
import pandas as pd
import numpy as np


# Function to check cosine similarity and generate labels
def check_similarity(weights, block_index, head_indices):
    num_heads = len(weights)
    tmp_dict = {}
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            v_i = F.normalize(weights[i].reshape(1, -1), p=2, dim=1)
            v_j = F.normalize(weights[j].reshape(1, -1), p=2, dim=1)
            cosine_sim = F.cosine_similarity(v_i, v_j)
            tmp_dict[(block_index, head_indices[i], head_indices[j])] = cosine_sim.item()
            #similarities.append(cosine_sim.item())
            #labels.append((block_index, head_indices[i], head_indices[j]))
    return tmp_dict


def plot_histogram(sim_dict_pt, sim_dict_ft, title, filename):
    # Extract similarity values and labels from the dictionaries
    similarities_pt = list(sim_dict_pt.values())
    similarities_ft = list(sim_dict_ft.values())
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    
    # Plot for pre-trained model
    #counts_pt, bins_pt, patches_pt = plt.hist(similarities_pt, bins=30, edgecolor='gray', color='red', alpha=0.5, label='Pre-trained')
    
    # Plot for fine-tuned model
    counts_ft, bins_ft, patches_ft = plt.hist(similarities_ft, bins=30, edgecolor='gray', color='skyblue', alpha=0.5, label='Fine-tuned')
    
    plt.legend(loc='upper right')
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory





# Main function to load the model and process weights
def main():
    ########## dir 
    save_directory = '../histograms/tiny/q'
    ############
    os.makedirs(save_directory, exist_ok=True)
    c_date, c_time = datetime.now().strftime("%m%d/%H%M%S").split('/')

    # pt vit
    model_pt = create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=True, num_classes=0)
    # model = create_model('vit_base_patch14_dinov2', pretrained=False)
    # model_pt = create_model('vit_base_patch16_224', pretrained=True)
    
    # ft vit
    model_ft = create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=False, num_classes=0)
    # model_ft = create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    # ckpt_path =  '/hdd/wi/sscd-copy-detection/ckpt/[baseog100][model_3]_OFFL_VIT_TINY_last_false_cls+pat_384_1107_062800/epoch=99-step=19499.ckpt'
    ckpt_path = "/hdd/wi/sscd-copy-detection/ckpt/exp_test/allortho_onlycont[model_3]_OFFL_VIT_TINY_last_True_0.001_cls+pat_192_1109_124028/qjnpx2vh/checkpoints/epoch=29-step=4679.ckpt"
    checkpoint = torch.load(ckpt_path)

    # # List of keys to be removed from the checkpoint state dictionary
    keys_to_remove = [
        "model.last_fc_layer.weight",
        "model.last_fc_layer.bias",
        "model.bn.weight",
        "model.bn.bias",
        "model.bn.running_mean",
        "model.bn.running_var",
        "model.bn.num_batches_tracked"
    ]

    # Remove the specified keys
    for key in keys_to_remove:
        if key in checkpoint['state_dict']:
            del checkpoint['state_dict'][key]

    adjusted_state_dict = {k.replace('model.backbone.', ''): v for k, v in checkpoint['state_dict'].items()}
    model_ft.load_state_dict(adjusted_state_dict, strict=True)

    model_pt.eval()
    model_ft.eval()
    
    sim_dict_pt = {}
    sim_dict_ft = {}

    # Iterate over all named parameters and find 'qkv.weight'
    for name, param in model_pt.named_parameters():
        if 'qkv.weight' in name:
            block_index = int(name.split('.')[1])  # Extract block index from parameter name
            wq_all, wk_all, wv_all = param.chunk(3, dim=0)
            w_vq = wq_all.chunk(3, dim=0)  # Chunk the value weights for each head
            w_vk = wk_all.chunk(3, dim=0)  # Chunk the value weights for each head
            w_vs = wv_all.chunk(3, dim=0)  # Chunk the value weights for each head
            
            # Generate head indices based on the number of heads
            head_indices = list(range(len(w_vq)))
            
            # Check similarity of V weights for each head and generate labels
            tmp_dict = check_similarity(w_vq, block_index, head_indices)
            # total_similarities.extend(similarities)
            # total_labels.extend(labels)
            sim_dict_pt.update(tmp_dict)
        # Iterate over all named parameters and find 'qkv.weight'
    del(tmp_dict)
    for name, param in model_ft.named_parameters():
        if 'qkv.weight' in name:
            block_index = int(name.split('.')[1])  # Extract block index from parameter name
            wq_all, wk_all, wv_all = param.chunk(3, dim=0)
            w_vq = wq_all.chunk(3, dim=0)  # Chunk the value weights for each head
            w_vk = wk_all.chunk(3, dim=0)  # Chunk the value weights for each head
            w_vs = wv_all.chunk(3, dim=0)  # Chunk the value weights for each head
            
            # Generate head indices based on the number of heads
            head_indices = list(range(len(w_vq)))
            
            # Check similarity of V weights for each head and generate labels
            tmp_dict = check_similarity(w_vq, block_index, head_indices)
            # total_similarities.extend(similarities)
            # total_labels.extend(labels)
            sim_dict_ft.update(tmp_dict)

    # Define the filename for the histogram with table
    filename = os.path.join(save_directory, f'cosine_similarity_table_{c_date}_{c_time}.png')
    # Save histogram of similarities with a table
    print(f"sim_dict_pt len is {len(sim_dict_pt)}")
    print(f"sim_dict_ft len is {len(sim_dict_ft)}")

    plot_histogram(sim_dict_pt, sim_dict_ft, 'Cosine Similarity Comparison', filename)
    print(f'Histogram with table saved as {filename}')

if __name__ == "__main__":
    main()
