# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#230804
from collections import OrderedDict 

import argparse
import enum
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, resnet50, resnext101_32x8d
from classy_vision.models import build_model
from .gem_pooling import GlobalGeMPool2d

from sscd.models import mae_vit
from sscd.models import dino_vit
from sscd.models import dtop_vit_384
from sscd.models import dtop_vit_192
from sscd.models import xcit

import timm
from timm.models import create_model

class Implementation(enum.Enum):
    CLASSY_VISION = enum.auto()
    TORCHVISION = enum.auto()
    TORCHVISION_ISC = enum.auto()
    OFFICIAL = enum.auto() ##230708##
    MOBILE = enum.auto() ##230708##
    MY = enum.auto() ##230925##
    
class Backbone(enum.Enum):
    CV_RESNET18 = ("resnet18", 512, Implementation.CLASSY_VISION)
    CV_RESNET50 = ("resnet50", 2048, Implementation.CLASSY_VISION)
    CV_RESNEXT101 = ("resnext101_32x4d", 2048, Implementation.CLASSY_VISION)

    TV_RESNET18 = (resnet18, 512, Implementation.TORCHVISION)
    TV_RESNET50 = (resnet50, 2048, Implementation.TORCHVISION)
    TV_RESNEXT101 = (resnext101_32x8d, 2048, Implementation.TORCHVISION)
    
    MULTI_RESNET50 = ("multigrain_resnet50", 2048, Implementation.TORCHVISION_ISC)

    OFFL_VIT = ('vit_patch_16_base', 768, Implementation.OFFICIAL)     ##230831#    
    OFFL_VIT_TINY = ('vit_patch_16_tiny', 192, Implementation.OFFICIAL)     ##230831#    
    OFFL_HYVIT_TINY = ('vit_tiny_r_s16_p8_224', 192, Implementation.OFFICIAL)

    OFFL_FAST_TINY_T8 = ('fastvit_t8', 192, Implementation.OFFICIAL)
    OFFL_FAST_TINY_T12 = ('fastvit_t12', 192, Implementation.OFFICIAL)
    OFFL_FAST_TINY_SA12 = ('fastvit_sa12', 192, Implementation.OFFICIAL)

    OFFL_DINO = ('dino_patch_16_base', 768, Implementation.OFFICIAL)     ##230708##
    OFFL_MAE = ('mae_patch_16_base', 768, Implementation.OFFICIAL)  
    OFFL_MOBVIT = ('mobilevit_xxs', 192, Implementation.MOBILE)
    
    MY_DTOP_VIT_192 = ('dtop_vit_tiny_192', 192, Implementation.MY)  
    MY_DTOP_VIT_384 = ('dtop_vit_tiny_384', 384, Implementation.MY)  
    MY_XCIT = ('xcit_retrievalv2_small_12_p16', 384, Implementation.MY)  

    MY_ORTHO_VIT = ('vit_patch_16_tiny', 192, Implementation.MY)  
    # MY_XCIT = ('vit_patch_16_small', 384, Implementation.MY)  
    # dino = ('dino_patch_16_base', 768, Implementation.OFFICIAL)     ##230708##
    
    def build(self, dims: int):
        impl = self.value[2]
        
        # print(self.value) #('resnet50', 2048, <Implementation.CLASSY_VISION: 1>)
        if impl == Implementation.CLASSY_VISION:
            model = build_model({"name": self.value[0]})
            # Remove head exec wrapper, which we don't need, and breaks pickling
            # (needed for spawn dataloaders).
            return model.classy_model
        
        if impl == Implementation.TORCHVISION:
            return self.value[0](num_classes=dims, zero_init_residual=True)
        # multigrain 230804
        
        if impl == Implementation.TORCHVISION_ISC:
            model = resnet50(pretrained=False)
            st = torch.load("/hdd/wi/isc2021/models/multigrain_joint_3B_0.5.pth")
            state_dict = OrderedDict([
                (name[9:], v)
                for name, v in st["model_state"].items() if name.startswith("features.")
            ])
            model.avgpool = nn.Identity()     
            model.fc = nn.Identity()
            # model.avgpool = None # None으로 하면 forward에서 호출하는 게 none이어서 
            # model.fc = None
            model.load_state_dict(state_dict, strict=True)
            return model

        if impl == Implementation.OFFICIAL: #### modi 0722 ###
            if self.value[0] == "vit_patch_16_base":
                model = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True, num_classes=0)
                return model           
            elif self.value[0] == "vit_patch_16_tiny":
                # model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=0)
                model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=0)
                return model            
            elif self.value[0] == "vit_tiny_r_s16_p8_224":
                model = timm.create_model("vit_tiny_r_s16_p8_224", pretrained=True, num_classes=0)
                return model
            elif self.value[0] == "fastvit_t8":
                model = timm.create_model("fastvit_t8", pretrained=True, num_classes=0)
                return model
            elif self.value[0] == "fastvit_t12":
                model = timm.create_model("fastvit_t12", pretrained=True, num_classes=0)
                return model
            elif self.value[0] == "fastvit_sa12":
                model = timm.create_model("fastvit_sa12", pretrained=True, num_classes=0)
                return model

            # elif self.value[0] == "dino_patch_16_base":
            #     model = dino_vit.__dict__['vit_base'](patch_size=16, num_classes=0)
            #     ckpt = torch.load("/hdd/wi/isc2021/models/dino_vitbase16_pretrain.pth", map_location=torch.device('cpu'))
            #     # new_ckpt = OrderedDict(("backbone."+k, v) for k, v in ckpt.items())
            #     # model.load_state_dict(ckpt, strict=True)
            #     print(model)
            #     print(f"===="*30)
            #     print(f"current state is")
            #     print(f"{ckpt.keys()}\sn")
            #     print(f"===="*30)
            #     print(f"Model {self.value[0]} built.")
            #     return model

        if impl == Implementation.MY:
            if self.value[0] == "dtop_vit_tiny_192":
                model = dtop_vit_192.DeepTokenPooling().cuda()
                return model
            
            if self.value[0] == "dtop_vit_tiny_384":
                model = dtop_vit_384.DeepTokenPooling().cuda()
                return model        
            
            # if self.value[0] == "xcit_retrievalv2_small_12_p16":

  
            #     model = create_model(
            #         'xcit_tiny_12_p16_224',
            #         num_classes=0,
            #         drop_rate=0.0,
            #         drop_path_rate=0.1,
            #         drop_block_rate=None
            #     )


            #     # checkpoint = torch.load('/hdd/wi/sscd-copy-detection/ckpt/xcit_small_12_p16_224.pth', map_location='cpu')

            #     # checkpoint_model = checkpoint['model']
            #     # state_dict = model.state_dict()
            #     # for k in ['head.weight', 'head.bias']:
            #     #     if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            #     #         print(f"Removing key {k} from pretrained checkpoint")
            #     #         del checkpoint_model[k]

            #     # model.load_state_dict(checkpoint_model, strict=False)
            #     model.cuda()
            #     print(model)
            #     return model
            
             
        else:
            raise AssertionError("Unsupported OFFICIAL model: %s" % (self.value[0]))
        

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)

class Model(nn.Module):
    def __init__(self, backbone: str, dims: int, pool_param: float): # og
    # def __init__(self, backbone: str, dims: int, pool_param: float, model_idx:int, token:str): # modi for mode
        super().__init__()
        self.backbone_type = Backbone[backbone] # self.backbone_type = <Backbone.CV_RESNET50>
                                                #<Backbone.CV_RESNET50: ('resnet50', 2048, <Implementation.CLASSY_VISION: 1>)>
                                                #Backbone <enum 'Backbone'> // 'CV_RESNET50'
        # MODI
        print(f"backbone is {backbone}")
        # self.model_idx = model_idx
        self.dims = self.backbone_type.value[1]
        self.backbone = self.backbone_type.build(dims=dims)
        impl = self.backbone_type.value[2]
        if impl == Implementation.CLASSY_VISION:
            self.embeddings = nn.Sequential(
                GlobalGeMPool2d(pool_param),
                nn.Linear(self.backbone_type.value[1], dims),
                L2Norm(),
            )
        elif impl == Implementation.TORCHVISION:
            if pool_param > 1:
                self.backbone.avgpool = GlobalGeMPool2d(pooling_param=3.0)
                fc = self.backbone.fc
                nn.init.xavier_uniform_(fc.weight)
                nn.init.constant_(fc.bias, 0)
            self.embeddings = L2Norm()
            # self.embeddings = nn.Identity()
        
        ## MODIFIED 230804##
        elif impl == Implementation.TORCHVISION_ISC:
            if pool_param > 1:
                self.backbone.avgpool = GlobalGeMPool2d(pooling_param=3.0)
                self.backbone.fc = nn.Linear(self.backbone_type.value[1], dims)
            self.embeddings = L2Norm()
        #classy vision은 모델에 pooling, fc없는데 torch vision이랑 pooling이 avg로 달려있음
            # self.embeddings = nn.Sequential(
            #     GlobalGeMPool2d(pooling_param=3.0),
            #     L2Norm(),
            # )
        ## MODIFIED 230724##
        elif impl == Implementation.OFFICIAL:
            if self.backbone_type.value[0] == "vit_patch_16_base":
                self.embeddings = L2Norm() 
            elif self.backbone_type.value[0] == "vit_patch_16_tiny":
                self.backbone.head_drop = nn.Identity()
                # self.dropout = nn.Dropout(p=0.5)
                # self.last_fc_layer = nn.Linear(self.dims*2, self.dims)  # 차원 축소 레이어 추가
                # self.bn = nn.BatchNorm1d(self.dims)
                # self.relu = nn.ReLU()
                self.embeddings = L2Norm()
            elif self.backbone_type.value[0] == "vit_tiny_r_s16_p8_224":
                self.embeddings = L2Norm()    
            # elif self.backbone_type.value[0] == "fastvit_t8":
            #     self.backbone.head_drop = nn.Identity()
            #     self.pooling = nn.AdaptiveAvgPool2d(1)
            #     self.fc_layer = nn.Linear(768, 192)
            #     self.embeddings = L2Norm()
            # elif self.backbone_type.value[0] == "fastvit_t12":
            #     self.backbone.head_drop = nn.Identity()
            #     self.pooling = nn.AdaptiveAvgPool2d(1)
            #     self.fc_layer = nn.Linear(1024, 192)
            #     self.embeddings = L2Norm()        
            # elif self.backbone_type.value[0] == "fastvit_sa12":
            #     self.backbone.head_drop = nn.Identity()
            #     self.pooling = nn.AdaptiveAvgPool2d(1)
            #     self.fc_layer = nn.Linear(1024, 192)
            #     self.embeddings = L2Norm()    
         

        elif impl == Implementation.MOBILE:
            self.embeddings = L2Norm()
        
        elif impl == Implementation.MY:
            self.backbone.head_drop = nn.Identity()
            self.dropout = nn.Dropout(p=0.5)
            # self.last_fc_layer = nn.Linear(self.dims*2, self.dims)  # 차원 축소 레이어 추가
            self.bn = nn.BatchNorm1d(self.dims)
            self.relu = nn.ReLU()
            self.embeddings = L2Norm()


    def forward(self, x): # 배치당 처리 
        ##################################################################
        # # [model 3] vit OG &&last blk && cls+pat concat 
        features = self.backbone.forward_features(x)
        cls_x = features[:, 0]
        pat_x = features[:, 1:].mean(dim=1)
        x = torch.cat([cls_x, pat_x], dim=1)
    
        ##################################################################
        # old code
        ##################################################################

        # # [model 1] vit OG && last blk && cls token
        # if self.model_idx == 1:
        #     x = self.backbone.get_global_feat(x)
        
        # # [model 2] vit OG && last blk && patch embedding
        # elif self.model_idx == 2:
        #     x = self.backbone.forward_features(x)[:, 1:].mean(dim=1)

        # # [model 3] vit OG &&last blk && cls+pat concat 
        # elif self.model_idx == 3:
        #     features = self.backbone.forward_features(x)
        #     cls_x = features[:, 0]
        #     pat_x = features[:, 1:].mean(dim=1)
        #     x = torch.cat([cls_x, pat_x], dim=1)

        # # [model 4] vit OG &&last blk && cls+pat linear
        # elif self.model_idx == 4:
        #     features = self.backbone.forward_features(x)
        #     cls_x = features[:, 0]
        #     pat_x = features[:, 1:].mean(dim=1)
        #     combined_features = torch.cat([cls_x, pat_x], dim=1)
        #     x = self.dropout(combined_features)
        #     x = self.last_fc_layer(x)
        #     x = self.bn(x)
        #     x = self.relu(x)
        
        # ##################################################################

        # # [model 5] Dtop and global branch
        # elif self.model_idx == 5:
        #     x = self.backbone(x)[0]

        # # [model 6] Dtop and local branch
        # elif self.model_idx == 6:
        #     x = self.backbone(x)[1]
        
        # # [model 7] Dtop and global && local branch concat 384dim
        # elif self.model_idx == 7:
        #     x = self.backbone(x)[2]
            
        # # [model 8] Dtop and global && local branch linear 192dim
        # elif self.model_idx == 8:
        #     x = self.backbone(x)[2]
        #     x = self.dropout(x)
        #     x = self.last_fc_layer(x)
        #     x = self.bn(x)
        #     x = self.relu(x)
            
        # ##################################################################
        
        # # [model 9] Dtop and global && local branch linear 384dim
        # elif self.model_idx == 9:
        #     x = self.backbone(x)
        #     x = self.dropout(x)
        #     x = self.last_fc_layer(x)
        #     x = self.bn(x)
        #     x = self.relu(x)
        
        # # [model 10] Dtop and global && local branch concat 384dim
        # elif self.model_idx == 10:
        #     x = self.backbone(x)[2]

            
            
        # return x
        return self.embeddings(x)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser = parser.add_argument_group("Model")
        parser.add_argument(
            "--backbone", default="TV_RESNET50", choices=[b.name for b in Backbone]            
        )
        parser.add_argument("--dims", default=512, type=int)
        parser.add_argument("--pool_param", default=3, type=float)
