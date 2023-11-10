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


def plot_histogram(sim_dict, title, filename):
    # Extract similarity values and labels from the dictionary
    similarities = list(sim_dict.values())
    labels = list(sim_dict.keys())
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(similarities, bins=30, edgecolor='gray')
    
    # Annotate each bin with labels
    bin_width = bins[1] - bins[0]
    label_positions = np.zeros_like(counts)
    
    for label, similarity in sim_dict.items():
        # Find the bin index for this similarity
        bin_index = np.digitize(similarity, bins) - 1
        # Make sure we don't go out of bounds
        bin_index = min(bin_index, len(counts) - 1)
        
        # Calculate the x and y positions for the label
        bin_center = bins[bin_index] + bin_width / 2
        label_y_position = label_positions[bin_index]
        
        # Annotate the bin with the label
        plt.text(bin_center, counts[bin_index] + label_y_position, str(label), ha='center', va='bottom', fontsize=6, rotation=60)
        
        # Update the label position for the next label in the same bin
        label_positions[bin_index] += counts[bin_index] * 0.1  # Increase by 10% of the bin height for each label

    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory


# Main function to load the model and process weights
def main():
    save_directory = '../histograms/que'
    os.makedirs(save_directory, exist_ok=True)
    c_date, c_time = datetime.now().strftime("%m%d/%H%M%S").split('/')

    # pt vit
    # model = create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=True)
    # model = create_model('vit_base_patch14_dinov2', pretrained=False)
    # model = create_model('vit_base_patch16_224', pretrained=False)
    
    # ft vit
    #  model = create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=False, num_classes=0)
    model = create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
    ckpt_path =  '/hdd/wi/sscd-copy-detection/ckpt/[baseog100][model_3]_OFFL_VIT_TINY_last_false_cls+pat_384_1107_062800/epoch=99-step=19499.ckpt'
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
    model.load_state_dict(adjusted_state_dict, strict=True)

    model.eval()
    
    sim_dict = {}

    # Iterate over all named parameters and find 'qkv.weight'
    for name, param in model.named_parameters():
        if 'qkv.weight' in name:
            block_index = int(name.split('.')[1])  # Extract block index from parameter name
            wq_all, wk_all, wv_all = param.chunk(3, dim=0)
            w_vq = wq_all.chunk(12, dim=0)  # Chunk the value weights for each head
            w_vk = wk_all.chunk(12, dim=0)  # Chunk the value weights for each head
            w_vs = wv_all.chunk(12, dim=0)  # Chunk the value weights for each head
            
            # Generate head indices based on the number of heads
            head_indices = list(range(len(w_vq)))
            
            # Check similarity of V weights for each head and generate labels
            tmp_dict = check_similarity(w_vq, block_index, head_indices)
            # total_similarities.extend(similarities)
            # total_labels.extend(labels)
            sim_dict.update(tmp_dict)

    # Define the filename for the histogram with table
    filename = os.path.join(save_directory, f'cosine_similarity_table_{c_date}_{c_time}.png')
    print(f"sim_dict is {sim_dict}")
    # Save histogram of similarities with a table
    plot_histogram(sim_dict, 'Cosine Similarity for all blocks', filename)
    print(f'Histogram with table saved as {filename}')

if __name__ == "__main__":
    main()
