from torch import nn
from torchvision import models

class UNetNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg16 =  models.vgg16(pretrained=True).features
        a=2

    def forward(self, input):

        out = self.vgg16(input)
        out1 = self.vgg16(input).features


        a=2

        return out
