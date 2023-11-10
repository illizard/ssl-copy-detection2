#################
# # modi ver
#################
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import torchextractor as tx

import timm
from timm.models.vision_transformer import VisionTransformer, Block

#1
class IRB(nn.Module):
    def __init__(self, in_channels=192):
        super(IRB, self).__init__()
        self.expand_ratio = 2
        hidden_dim = round(in_channels * self.expand_ratio)
        self.use_res_connect = self.expand_ratio == 1

        layers = []
        if self.expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

# class WaveBlock(nn.Module):
#     def __init__(self, rw=0.3, rh=1):
#         super(WaveBlock, self).__init__()
#         self.rw = rw
#         self.rh = rh

#     def forward(self, x):
#         # x의 shape: [batch, layer, channel, height, width] = [4, 1, 768, 14, 14]
#         B, L, C, H, W = x.size()
        
#         # 높이에 대한 랜덤 값을 생성합니다.
#         X = random.randint(0, int(H * (1 - self.rw)))
        
#         # 랜덤 값에 따라 feature map을 수정합니다.
#         mask = torch.ones_like(x)
#         mask[:, :, :, X:X + int(H * self.rw), :] = self.rh
#         x_wave = x * mask

#         return x_wave

#2
class ASPP(nn.Module):
    def __init__(self, in_channels=192, out_channels=384):
        super(ASPP, self).__init__()
        
        # 1x1 convolution branch
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(out_channels)
        
        # Atrous convolutions with different dilation rates
        self.atrous_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.bn_atrous_conv1 = nn.BatchNorm2d(out_channels)
        
        self.atrous_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.bn_atrous_conv2 = nn.BatchNorm2d(out_channels)

        self.atrous_conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False)
        self.bn_atrous_conv3 = nn.BatchNorm2d(out_channels)

        # Adaptive average pooling branch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution to reduce the channel dimension after concatenation
        self.output_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_output_conv = nn.BatchNorm2d(out_channels)

        # Final convolution to get to num_classes
        # self.conv_1x1_4 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_1x1_4 = nn.Conv2d(out_channels, in_channels, kernel_size=1)

        self.relu = nn.ReLU()
        
    def forward(self, x): # x.shape torch.Size([20, 192, 14, 14])
        
        x1 = self.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x)))
        x2 = self.relu(self.bn_atrous_conv1(self.atrous_conv1(x)))
        x3 = self.relu(self.bn_atrous_conv2(self.atrous_conv2(x)))
        x4 = self.relu(self.bn_atrous_conv3(self.atrous_conv3(x)))
        
        # Adaptive average pooling branch
        x5 = self.avg_pool(x)
        x5 = self.relu(self.bn_conv_1x1_2(self.conv_1x1_2(x5)))
        x5 = F.interpolate(x5, size=x.size()[2:], mode="bilinear", align_corners=True)
        
        # Concatenate along the channel dimension
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.relu(self.bn_output_conv(self.output_conv(x)))
        
        return self.conv_1x1_4(x)
    
class EnhancedLocalityModule(nn.Module):
    def __init__(self, in_channels):
        super(EnhancedLocalityModule, self).__init__()
        self.irb = IRB(in_channels)
        self.aspp = ASPP()

    def forward(self, x):
        x = self.irb(x)
        x = self.aspp(x)
        # print(f"x is {x.shape}")

        return x



'''
F_c_dim = class token embedding dimension
'''

    
# class GlobalBranch(nn.Module):
#     def __init__(self, F_c_dim): #torch.Size([4, 12, 192]) , F_c_dim =192
#         super(GlobalBranch, self).__init__()
#         #layer 6
#         # self.fc = nn.Linear(12, 1)
#         self.fc = nn.Linear(6, 1)
#         self.relu = nn.ReLU()
#         self.F_c_dim = F_c_dim
        
#     def forward(self, F_c):
#         x = self.relu(self.fc(F_c.transpose(1, 2)))
#         u_c = x.transpose(1, 2) 
        
#         return u_c  #torch.Size([4, 1, 192])
    
class GlobalBranch(nn.Module):
    def __init__(self, F_c_dim): #torch.Size([4, 12, 192]) , F_c_dim =192
        super(GlobalBranch, self).__init__()
        self.F_c_dim = F_c_dim
        #layer 6
        # self.fc = nn.Linear(12, 1)
        self.fc1 = nn.Linear(6, 1)
        self.fc2 = nn.Linear(self.F_c_dim, self.F_c_dim*2)
        self.relu = nn.ReLU()
        self.F_c_dim = F_c_dim
        
    def forward(self, F_c):
        x = self.relu(self.fc1(F_c.squeeze(dim=2).transpose(1, 2)))
        x = x.transpose(1, 2)
        u_c = self.relu(self.fc2(x))  # 마지막 차원을 384로 확장

        return u_c  #torch.Size([4, 1, 384])
        

# Local Branch
class LocalBranch(nn.Module):
    def __init__(self, k, F_p_dim):
        super(LocalBranch, self).__init__()
        self.k = k # k=12
        self.wh = 196
        self.out_channels = 1
        self.F_p_dim = F_p_dim #192
        # self.conv1x1 = nn.Conv2d(self.k, self.out_channels, kernel_size=1)  # Added 1x1 convolution layer.
        self.conv1x1 = nn.Conv3d(self.k, self.out_channels, kernel_size=(1, 1, 1))  # 3D convolution layer
        self.elm = EnhancedLocalityModule(self.F_p_dim)  # Uncommented this line to use ELM.
        self.fc = nn.Linear(self.F_p_dim, self.F_p_dim*2)
        
    def forward(self, x):   #x #torch.Size([20, 12, 196, 192])
        
        # x = x.permute(0, 1, 3, 2).contiguous().view(x.size(0), x.size(1), x.size(-1), int((x.size(2))**0.5), int((x.size(2))**0.5)) # 3d tensor unfold
        # x = x.mean(dim=1) # =>torch.Size([20, 192, 14, 14])
        
        # 1. 3d tentsor unfold
        # og vit model
        x = x.permute(0, 1, 3, 2).contiguous().view(x.size(0), x.size(1), x.size(-1), 14, 14)
        # hybrid vit model 
        # x = x.permute(0, 1, 3, 2).contiguous().view(x.size(0), x.size(1), x.size(-1), int((x.size(2))**0.5), int((x.size(2))**0.5))
        
        # 2. 1x1x conv for kd -> d
        x = self.conv1x1(x) # Apply 1x1 convolution before ELM. #torch.Size([4, 1, 768, 14, 14])
        # Squeeze only if batch size is 1 or the extra channel dimension is 1
        if x.size(0) == int(1):
            x = x.squeeze(dim=1)
        if x.size(0) != int(1):
            x = x.squeeze()
        x_og = x.clone()
        
        # 3. enhanced locality module
        elm_x = self.elm(x)  # Apply ELM. 여기서 걸림
        x = x_og + elm_x # Fuse
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).permute(0, 2, 1).squeeze(-1).squeeze(-1)
        u_p = self.fc(x)

        return u_p #torch.Size([20, 384])

      
class DeepTokenPooling(nn.Module):
    def __init__(self):
        super(DeepTokenPooling, self).__init__()    
        # common
        # paper = D =768 원래 디멘션 ,  N = 1536
        
        # for global branch
        self.F_c_dim = 192
        self.N = self.F_c_dim * 2 # final descriptor dim (global + local) 
        #self.k = 12       
        self.k = 6       
        
        # Initialize self.features as None. We will allocate it dynamically.
        self.features = None
        # Initialize model
        self.global_branch = GlobalBranch(self.F_c_dim)
        self.local_branch = LocalBranch(self.k, self.F_c_dim)
        # for layer 6
        self.block_names = [f"blocks.{i}" for i in range(6, 12)]
        # self.block_names = [f"blocks.{i}" for i in range(12)]
        self.vit_backbone = timm.create_model("vit_tiny_patch16_224.augreg_in21k", num_classes=0, pretrained=True)
        # self.vit_backbone = timm.create_model("vit_tiny_r_s16_p8_224", num_classes=0, pretrained=True)

        # self.dropout = nn.Dropout(p=0.5)
        # self.last_fc_layer = nn.Linear(self.N, self.F_c_dim)  # 차원 축소 레이어 추가
        # self.bn = nn.BatchNorm1d(192)
        # self.relu = nn.ReLU()
 

    def forward(self, x):

        # 일단 현재까지 완성버전
        # layer
        all_feat_tuple = self.vit_backbone.get_intermediate_layers(x, n=6, norm=True, return_class_token=True) # => all_feat_tuple tuple(blks, pat/cls, batch, 196, 192)
        cls_tokens = []
        patch_embeddings = []

        for block in all_feat_tuple:
            patch_embedding, class_token = block
            cls_tokens.append(class_token)
            patch_embeddings.append(patch_embedding)

        cls_tokens = torch.stack(cls_tokens, dim=1)  # [batch, blks, 1, 192]
        patch_tokens = torch.stack(patch_embeddings, dim=1)  # [batch, blks, 196, 192]
        
        # ## [model 9] each feaures DTOP ver
        global_features = self.global_branch(cls_tokens)
        local_features = self.local_branch(patch_tokens)
        combined_features = torch.cat([global_features, local_features], dim=2).squeeze(dim=1)

        
        return combined_features #[batch, 768] [b, 2N] = [b, 4D] D=192, N = 384   

def controlRandomness(random_seed=42):

    if random_seed is not None:
        print(f"random seed = {random_seed}")
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # np.random.seed(random_seed)
        random.seed(random_seed)

    else: # random
        print(f"random seed = {random_seed}")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
    controlRandomness()
    # Create the custom model
    model = DeepTokenPooling()
    print(model)
    for param in model.parameters():
        print(param.requires_grad)

    dummy = torch.randn(1, 3, 224, 224)
    # combined_features, vit_features = model(dummy)
    vit_features = model(dummy)
    print(vit_features.shape)
    
