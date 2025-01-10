import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        
        # Adjust the input layer to accept 3 channels and the desired input size
        self.features[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Add a new fully connected layer
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

