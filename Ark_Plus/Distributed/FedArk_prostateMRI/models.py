import torch
import torch.nn as nn

from nets import DenseNet, UNet

class ArkUNet(UNet):
    def __init__(self, num_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.omni_heads = []
        for _ in range(num_head):
            self.omni_heads.append(
                nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
                nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1))
            )
        self.omni_heads = nn.ModuleList(self.omni_heads)

    def forward(self, x, head_n=None):
        bottleneck, x = self.forward_features(x)
        if head_n is not None:
            return bottleneck, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]
    
    # def generate_embeddings(self, x, after_proj = True):
    #     x = self.forward_features(x)
    #     if after_proj:
    #         x = self.projector(x)
    #     return x

class ArkDenseNet(DenseNet):
    def __init__(self, num_classes_list, projector_features = None, use_mlp=False, encoder_features=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None 
        if projector_features:
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(nn.Linear(encoder_features, self.num_features), nn.ReLU(inplace=True), nn.Linear(self.num_features, self.num_features))
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)

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

def build_omni_model(args, num_classes_list=None):
    if args.model_name == "densenet":
        model = ArkDenseNet(num_classes_list, args.projector_features, args.use_mlp, input_shape =(3, args.crop_size, args.crop_size))

    elif args.model_name == "unet":
        model = ArkUNet(num_head=6, input_shape=[3, 384, 384])
    return model

def save_checkpoint(state,filename='model'):

    torch.save(state, filename + '.pth.tar')
