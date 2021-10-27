import pytorch_lightning as pl
from ReadData import ReadData
import torch
import torch.nn as nn


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
        _, axs = plt.subplots(1, 3, figsize=(15,15))
        self.axs = axs

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def training_step(self, batch, batch_idx):
        out = self.model(batch[0])

        loss = self.criterion(out, batch[1])

        if batch_idx % 10 == 0:
            out0 = batch[0][0,:,:,:].permute(1,2,0).detach().cpu().numpy()
            out1 = batch[1][0,:,:,:].permute(1,2,0).detach().cpu().numpy()
            out2 = out[0,:,:,:].permute(1,2,0).detach().cpu().numpy()

            self.axs[0].imshow(out0.astype(np.float32), cmap="gray")
            self.axs[1].imshow(out1.astype(np.float32), cmap="gray")
            self.axs[2].imshow(out2.astype(np.float32), cmap="gray")
            plt.show()

        a = 2
        return loss



from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == "__main__":
    unetTrainer = UNetTrainer()

    trainer = pl.Trainer(
        gpus = min(1, torch.cuda.device_count()),
        max_epochs=200,
        limit_val_batches=2,
        precision=16,
        log_every_n_steps=40,
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
    

    train_dataloader = DataLoader(ReadData(transform, transform1))

    trainer.fit(unetTrainer, train_dataloader)
