import pytorch_lightning as pl
from ReadData import ReadData
import torch
import torch.nn as nn
from torchvision.utils import save_image


from UNet import UNetNetwork

import matplotlib.pyplot as plt
import numpy as np

#Contrast Stretch
def ContrastStretch(f, K=1):
    fMin = np.min(f, axis=(0,1))
    fMax = np.max(f, axis=(0,1))

    fs = K * ((f - fMin) / (fMax-fMin))
    return fs

class UNetTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model =  UNetNetwork()
        self.criterion = nn.BCEWithLogitsLoss()
        self.L1 = nn.L1Loss()
        _, axs = plt.subplots(1, 3, figsize=(15,15))
        self.axs = axs

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def training_step(self, batch, batch_idx):
        out = self.model(batch[0])
        loss = self.criterion(out, batch[1])
        return loss

    def validation_step(self, batch, batch_idx):
        totalDone = self.current_epoch * len(self.train_dataloader()) + batch_idx

        out = self.model(batch[0])
        loss = self.criterion(out, batch[1])
        img_sample = torch.cat((batch[1].data, out.data), -1)
        save_image(img_sample, "Outputs/{0}.png".format(totalDone), nrow=1, normalize=True)

        return loss


    def test_step(self, batch, batch_idx):
        return 0


from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":
    unetTrainer = UNetTrainer()

    trainer = pl.Trainer(
        gpus = min(1, torch.cuda.device_count()),
        max_epochs=30,
        precision=16,
        logger=False,
        checkpoint_callback=False
    )

    transform = [
        #transforms.Resize((512, 1024), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    ]

    transform1 = [
        #transforms.Resize((512, 1024), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
    ]
    

    train_dataloader = DataLoader(ReadData('./Data/Training', transform, transform1, SplitDataSet = True))
    val_dataloader = DataLoader(ReadData('./Data/Validation', transform, transform1, SplitDataSet = False))
    test_dataloader = DataLoader(ReadData('./Data/Testing', transform, transform1, SplitDataSet = False))


    trainer.fit(unetTrainer, train_dataloader, val_dataloader)
