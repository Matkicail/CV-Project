import pytorch_lightning as pl
from ReadData import ReadData
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np

from UNet import UNetNetwork

class UNetTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        #Create Model
        self.model =  UNetNetwork()

        #Set up loss
        self.criterion = nn.BCEWithLogitsLoss()

    #Set up optimization step
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #Training Step
    def training_step(self, batch, batch_idx):
        out = self.model(batch[0])
        loss = self.criterion(out, batch[1])
        return loss

    #Validation Outputs
    def validation_step(self, batch, batch_idx):
        totalDone = self.current_epoch * len(self.train_dataloader()) + batch_idx

        out = self.model(batch[0])
        loss = self.criterion(out, batch[1])
        img_sample = torch.cat((batch[1].data, out.data), -1)
        save_image(img_sample, "Outputs/{0}.png".format(totalDone), nrow=1, normalize=True)

        return loss

    
    def test_step(self, batch, batch_idx):
        return 0

#--------------Start--------------#
from torch.utils.data import DataLoader                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
from torchvision import transforms

if __name__ == "__main__":
    unetTrainer = UNetTrainer()

    #Create Trainer
    trainer = pl.Trainer(
        gpus = min(1, torch.cuda.device_count()),
        max_epochs=30,
        precision=16,
    )

    #Transform for puzzle
    transform = [
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    ]

    #Transform for mask
    transform1 = [
        transforms.ToTensor(),
    ]
    
    #Create Dataloaders for feeding the network the images
    train_dataloader = DataLoader(ReadData('./Data/Training', transform, transform1, splitDataSet = False, augmentDataSet=False), num_workers=3)
    val_dataloader = DataLoader(ReadData('./Data/Validation', transform, transform1, splitDataSet = False), num_workers=3)
    test_dataloader = DataLoader(ReadData('./Data/Testing', transform, transform1, splitDataSet = False))

    #Fit
    trainer.fit(unetTrainer, train_dataloader, val_dataloader)
