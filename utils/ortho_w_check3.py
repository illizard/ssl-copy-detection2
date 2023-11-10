import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb

import timm 

def compute_weight_orthogonality():
    w_dict = {}
    model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=0)
    for name, param in model.named_parameters():

    # for name, param in self.model.named_parameters():
        if 'qkv.weight' in name:
            wq_all, wk_all, wv_all = param.chunk(3, dim=0)
            
            # Assuming 3 heads
            w_qs = wq_all.chunk(3, dim=0)
            w_ks = wk_all.chunk(3, dim=0) 
            w_vs = wv_all.chunk(3, dim=0)
            
            # Flatten the weights for each head
            flattened_q_weights = torch.stack([w_q_head.flatten() for w_q_head in w_qs]).requires_grad_(True)
            flattened_k_weights = torch.stack([w_k_head.flatten() for w_k_head in w_ks]).requires_grad_(True)
            flattened_v_weights = torch.stack([w_v_head.flatten() for w_v_head in w_vs]).requires_grad_(True)
            
            # Normalize the flattened weights to ensure the diagonal of the Gram matrix is 1
            normalized_q_weights = F.normalize(flattened_q_weights, p=2, dim=1).requires_grad_(True)
            normalized_k_weights = F.normalize(flattened_k_weights, p=2, dim=1).requires_grad_(True)
            normalized_v_weights = F.normalize(flattened_v_weights, p=2, dim=1).requires_grad_(True)
            
            # Compute the cosine similarity matrix
            wq_gram_matrix = torch.matmul(normalized_q_weights, normalized_q_weights.T).requires_grad_(True)
            wk_gram_matrix = torch.matmul(normalized_k_weights, normalized_k_weights.T).requires_grad_(True)
            wv_gram_matrix = torch.matmul(normalized_v_weights, normalized_v_weights.T).requires_grad_(True)
            pdb.set_trace()

            # Scale the off-diagonal elements to increase variance
            scale_factor = 1000  # This can be tuned
            # For the query weight matrix
            wq_gram_matrix_off_diagonal = wq_gram_matrix.clone()
            wq_gram_matrix_off_diagonal.fill_diagonal_(0).requires_grad_(True)
            wq_gram_matrix = wq_gram_matrix_off_diagonal * scale_factor + torch.diag_embed(torch.diag(wq_gram_matrix).requires_grad_(True)).requires_grad_(True)

            # For the key weight matrix
            wk_gram_matrix_off_diagonal = wk_gram_matrix.clone()
            wk_gram_matrix_off_diagonal.fill_diagonal_(0).requires_grad_(True)
            wk_gram_matrix = wk_gram_matrix_off_diagonal * scale_factor + torch.diag_embed(torch.diag(wk_gram_matrix).requires_grad_(True)).requires_grad_(True)

            # For the value weight matrix
            wv_gram_matrix_off_diagonal = wv_gram_matrix.clone()
            wv_gram_matrix_off_diagonal.fill_diagonal_(0).requires_grad_(True)
            wv_gram_matrix = wv_gram_matrix_off_diagonal * scale_factor + torch.diag_embed(torch.diag(wv_gram_matrix).requires_grad_(True)).requires_grad_(True)

            
            # Calculate the loss that encourages orthogonality using torch.square
            wq_orth_loss = torch.sum(torch.square(torch.triu(wq_gram_matrix, diagonal=1).requires_grad_(True)).requires_grad_(True)).requires_grad_(True)
            wk_orth_loss = torch.sum(torch.square(torch.triu(wk_gram_matrix, diagonal=1).requires_grad_(True)).requires_grad_(True)).requires_grad_(True)
            wv_orth_loss = torch.sum(torch.square(torch.triu(wv_gram_matrix, diagonal=1).requires_grad_(True)).requires_grad_(True)).requires_grad_(True)
            
            orth_loss = wq_orth_loss + wk_orth_loss + wv_orth_loss
            orth_loss.requires_grad_(True)
            w_dict[name] = orth_loss
            
    return w_dict


if __name__ == '__main__':
    # model.check_orthogonality()
    
    

    compute_weight_orthogonality()
    # vit_features = model(dummy)
    # print(model)
    # print(vit_features.shape)


    