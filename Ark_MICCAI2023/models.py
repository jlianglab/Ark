import torch
import torch.nn as nn
from functools import partial
from torch.hub import load_state_dict_from_url

import timm.models.vision_transformer as vit
import timm.models.swin_transformer as swin
# import timm.models.efficientnet as effinet
 
from timm.models.helpers import load_state_dict


class ArkSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)
        
        #multi-task heads
        self.omni_heads = []  
        for num_classes in num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj = True):
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x
            
def build_omni_model_from_checkpoint(args, num_classes_list, key):
    
    if args.model_name == "swin_base": #swin_base_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
    
    if args.pretrained_weights is not None:
        checkpoint = torch.load(args.pretrained_weights)
        state_dict = checkpoint[key]
        if any([True if 'module.' in k else False for k in state_dict.keys()]):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}

        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))     
           
    return model

def build_omni_model(args, num_classes_list):
    if args.model_name == "swin_base": #swin_base_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))

    if args.pretrained_weights is not None:
        if args.pretrained_weights.startswith('https'):
            state_dict = load_state_dict_from_url(url=args.pretrained_weights, map_location='cpu')
        else:
            state_dict = load_state_dict(args.pretrained_weights)
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
      
        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))

    return model

def save_checkpoint(state,filename='model'):
    torch.save(state, filename + '.pth.tar')