import torch
from torch import nn
from torchvision import models
from Blocks import Decoder

class UNetNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        #Load VGG Model
        vgg16 = models.vgg16(pretrained=True).features

        #Split for skip connections
        self.down1 = vgg16[0:5]
        self.down2 = vgg16[5:10]
        self.down3 = vgg16[10:17]
        self.down4 = vgg16[17:24]
        self.down5 = vgg16[24:31]

        #Create Decoder Side
        self.out1 = Decoder(512, 512)
        self.out2 = Decoder(2*512, 256)
        self.out3 = Decoder(2*256, 128)
        self.out4 = Decoder(2*128, 64)

        #Final Layer uses sigmoid activation
        self.out5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(2*64, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):

        #Encoder
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        #Decoder
        u1 = torch.utils.checkpoint.checkpoint(self.out1, d5, d4) #Gradient checkpointing
        u2 = self.out2(u1, d3)
        u3 = self.out3(u2, d2)
        u4 = self.out4(u3, d1)
        return self.out5(u4)
