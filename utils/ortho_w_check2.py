import torch
import torch.nn.functional as F
import torch.nn as nn
    
def compute_weight_orthogonality(self):
    w_dict = {}
    for name, param in self.model.named_parameters():
        if 'qkv.weight' in name:
            
            # value version
            # _, _, wv_all = param.chunk(3, dim=0)
            wq_all, wk_all, wv_all = param.chunk(3, dim=0)
            
            w_qs = wq_all.chunk(3, dim=0)  # Assuming 3 heads
            w_ks = wk_all.chunk(3, dim=0)  # Assuming 3 heads 
            w_vs = wv_all.chunk(3, dim=0)  # Assuming 3 heads
            flattened_q_weights = torch.stack([w_q_head.flatten() for w_q_head in w_qs]).requires_grad_(True)
            flattened_k_weights = torch.stack([w_k_head.flatten() for w_k_head in w_ks]).requires_grad_(True)
            flattened_v_weights = torch.stack([w_v_head.flatten() for w_v_head in w_vs]).requires_grad_(True)
        
            # Compute the cosine similarity matrix
            wq_gram_matrix = torch.matmul(flattened_q_weights, flattened_q_weights.T).requires_grad_(True)
            wq_gram_matrix_triu = torch.triu(wq_gram_matrix, diagonal=1).requires_grad_(True)
            wq_gram_matrix_triu_squared =  torch.square(wq_gram_matrix_triu).requires_grad_(True)
            q_orth_loss =  torch.sum(wq_gram_matrix_triu_squared).requires_grad_(True)
            
            # Compute the cosine similarity matrix
            wk_gram_matrix = torch.matmul(flattened_k_weights, flattened_k_weights.T).requires_grad_(True)
            wk_gram_matrix_triu = torch.triu(wk_gram_matrix, diagonal=1).requires_grad_(True)
            wk_gram_matrix_triu_squared =  torch.square(wk_gram_matrix_triu).requires_grad_(True)
            k_orth_loss =  torch.sum(wk_gram_matrix_triu_squared).requires_grad_(True)
            
            # Compute the cosine similarity matrix
            wv_gram_matrix = torch.matmul(flattened_v_weights, flattened_v_weights.T).requires_grad_(True)
            wv_gram_matrix_triu = torch.triu(wv_gram_matrix, diagonal=1).requires_grad_(True)
            wv_gram_matrix_triu_squared =  torch.square(wv_gram_matrix_triu).requires_grad_(True)
            v_orth_loss =  torch.sum(wv_gram_matrix_triu_squared).requires_grad_(True)

            # # Compute the cosine similarity matrix
            # w_gram_matrix = torch.matmul(flattened_weights, flattened_weights.T).requires_grad_(True)
            # w_gram_matrix_triu = torch.triu(w_gram_matrix, diagonal=1).requires_grad_(True)
            # w_gram_matrix_triu_squared =  torch.square(w_gram_matrix_triu).requires_grad_(True)
            # orth_loss = torch.sum(w_gram_matrix_triu_squared).requires_grad_(True)
            
            # orth_loss = torch.sum(v_orth_loss).requires_grad_(True)            
            orth_loss = q_orth_loss + k_orth_loss + v_orth_loss
            orth_loss.requires_grad_(True)
            w_dict[name] = orth_loss
            
    return w_dict

if __name__ == '__main__':
    # model = CustomViT()
    # model.check_orthogonality()
    dummy = torch.randn(4, 3, 224, 224)
    # vit_features = model(dummy)
    # print(model)
    # print(vit_features.shape)


    