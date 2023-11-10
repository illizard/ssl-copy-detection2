import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pdb
import timm 

def compute_loss(gram_matrix):
    diag_elements = torch.diag(gram_matrix)
    off_diag_elements = gram_matrix - torch.diag_embed(diag_elements)
    
    mean_off_diag = off_diag_elements.mean()
    var_off_diag = ((off_diag_elements - mean_off_diag) ** 2).mean()
    
    # We create a loss that has both a negative mean and a high variance for the off-diagonal elements
    # This encourages the off-diagonal elements to be negative while also spread out (high variance)
    loss = -(var_off_diag + mean_off_diag) # Minimizing this loss will maximize variance and minimize the mean
    return loss


# def compute_variance_loss(gram_matrix):
#     diag_elements = torch.diag(gram_matrix)
#     off_diag_elements = gram_matrix - torch.diag_embed(diag_elements)
#     # pdb.set_trace()
# # 
#     # We want to maximize the variance of the off-diagonal elements
#     mean_off_diag = off_diag_elements.mean()
#     var_off_diag = ((off_diag_elements - mean_off_diag) ** 2).mean()
#     # We negate the variance because we want to maximize it, hence minimizing the negative variance
#     loss = -var_off_diag
#     return loss

def compute_weight_orthogonality(self):
    w_dict = {}
    # model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=0)
    for name, param in self.model.named_parameters():

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
            # pdb.set_trace()
                        
    #         # Compute the variance loss for each Gram matrix
    #         loss_wq_var = compute_variance_loss(wq_gram_matrix)
    #         loss_wk_var = compute_variance_loss(wk_gram_matrix)
    #         loss_wv_var = compute_variance_loss(wv_gram_matrix)
            
    #         orth_loss = loss_wq_var + loss_wk_var + loss_wv_var

    #         w_dict[name] = orth_loss.requires_grad_(True)

    # return w_dict
            # Compute the combined loss for each Gram matrix
            loss_wq = compute_loss(wq_gram_matrix)
            loss_wk = compute_loss(wk_gram_matrix)
            loss_wv = compute_loss(wv_gram_matrix)
            
            # The total loss for orthogonality
            orth_loss = loss_wq + loss_wk + loss_wv

            w_dict[name] = orth_loss

    return w_dict
                
if __name__ == '__main__':
    compute_weight_orthogonality()
