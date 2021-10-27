from torch import nn
from torchvision import models
from Blocks import Decoder

class UNetNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg16 =  models.vgg16(pretrained=True).features

        self.out1 = Decoder(512, 512)
        self.out2 = Decoder(512, 256)
        self.out3 = Decoder(256, 128)
        self.out4 = Decoder(128, 64)
        #self.out5 = Decoder(64, 1)

        self.out5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(64, 1, 4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):

        #Encoder
        out = self.vgg16(input)

        out = self.out1(out)

        out = self.out2(out)

        out = self.out3(out)

        out = self.out4(out)

        out = self.out5(out)

        return out
