import torch
import torch.nn as nn
import torchvision


class VGG(nn.Module):

    def __init__(self, pretrained=True, weights_path=None, requires_grad=False):
        super().__init__()
        self.style_layers = [0, 5, 10, 19, 28]

        if pretrained:
            self.model = torchvision.models.vgg19(pretrained=True).features[:29]
        else:
            self.model = self.load_vgg19(weights_path).features[:29]
        
        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, x):
        style_features = []
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in self.style_layers:
                style_features.append(x)
        return style_features


    @staticmethod
    def load_vgg19(weights_path):
        model = torchvision.models.vgg19(weights=None)
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path))
        return model