import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from ReadData import ReadData
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np

from UNet import UNetNetwork
from torch.utils.tensorboard import SummaryWriter

class UNetTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        #Create Model
        self.model =  UNetNetwork()

        #Set up loss
        self.criterion = nn.BCEWithLogitsLoss()
        self.L1_criterion = nn.L1Loss()


    #Set up optimization step
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #Training Step
    def training_step(self, batch, batch_idx):
        out = self.model(batch[0])
        loss = self.criterion(out, batch[1])
        self.log("Loss/Train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    #Validation Outputs
    def validation_step(self, batch, batch_idx):
        totalDone = self.current_epoch * len(self.train_dataloader()) + batch_idx

        out = self.model(batch[0])
        loss = self.criterion(out, batch[1])
        img_sample = torch.cat((batch[1].data, out.data), -1)
        save_image(img_sample, "Outputs/Validation/{0}.png".format(totalDone), nrow=1, normalize=True)
        self.log("Loss/Validation", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    
    def test_step(self, batch, batch_idx):
        totalDone = self.current_epoch * len(self.test_dataloader()) + batch_idx

        out = self.model(batch[0])
        loss = self.criterion(out, batch[1])
        lossL1 = self.L1_criterion(out, batch[1])
        img_sample = torch.cat((batch[1].data, out.data), -1)
        save_image(img_sample, "Outputs/Test/{0}.png".format(totalDone), nrow=1, normalize=True)
        self.log("Loss/Test_BCE", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("Loss/Test_L1", lossL1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


#--------------Start--------------#
from torch.utils.data import DataLoader                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
from torchvision import transforms

if __name__ == "__main__":
    unetTrainer = UNetTrainer()

    tb_logger = pl_loggers.TensorBoardLogger("logs/")

    #Create Trainer
    trainer = pl.Trainer(
        gpus = min(1, torch.cuda.device_count()),
        max_epochs=8,
        precision=16,
        logger=tb_logger,
        log_every_n_steps=1
    )

    #Transform for puzzle
    transform = [
        transforms.ToTensor(),
    ]

    #Create Dataloaders for feeding the network the images
    train_dataloader = DataLoader(ReadData('./Data/Training', transform, splitDataSet = False, augmentDataSet=False), num_workers=3)
    val_dataloader = DataLoader(ReadData('./Data/Validation', transform, splitDataSet = False), num_workers=3)
    test_dataloader = DataLoader(ReadData('./Data/Test', transform, splitDataSet = False))

    #Fit
    trainer.fit(unetTrainer, train_dataloader, val_dataloader)
    trainer.test(unetTrainer, test_dataloader)
