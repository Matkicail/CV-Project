from torch import nn
import torch

class Decoder(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Decoder, self).__init__()

        layers = [
            #n.ConvTranspose2d(in_size, out_size, 4, 2, 1, output_padding=(0,0), bias=False),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=0),
            nn.GroupNorm(2, out_size, affine=True),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
    
    def forward(self, input, skip_input=None):
        #output = None
        output = self.model(input)
        #output = torch.cat((output, skip_input), 1)

        return output