from torch import nn
import torch

class Decoder(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Decoder, self).__init__()

        #Layers used in the decoder block
        layers = [
            #n.ConvTranspose2d(in_size, out_size, 4, 2, 1, output_padding=(0,0), bias=False),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        
        #Drop out if wanted
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    #Pass forward    
    def forward(self, input, skip_input):
        output = self.model(input)
        output = torch.cat((output, skip_input), 1)

        return output