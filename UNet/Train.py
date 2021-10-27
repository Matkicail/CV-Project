import pytorch_lightning as pl
from ReadData import ReadData
import torch

from UNet import UNetNetwork

class UNetTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model =  UNetNetwork()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def training_step(self, batch, batch_idx):
        out = self.model(batch[0])

        a = 2



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
    

    train_dataloader = DataLoader(ReadData(transform))

    trainer.fit(unetTrainer, train_dataloader)
