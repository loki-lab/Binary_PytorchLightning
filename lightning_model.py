from torch import nn
from model import VGG16
from pytorch_lightning import LightningModule
from torchmetrics.classification import Accuracy
from torch.optim import Adam
from torch.utils.data import DataLoader

class VGG16Lightning(LightningModule):
    def __init__(self, num_classes, train_ds, val_ds):
        super().__init__()
        self.model = VGG16(num_classes=num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_ds = train_ds 
        self.val_ds = val_ds
    
    def forward(self, x):
        out = self.model(x)
        return out
    
    def training_step(self, input, input_idx):
        image, label = input
        output = self.forward(image)
        loss = nn.CrossEntropyLoss()(output, label)
        accuracy = self.accuracy(output, label)
        self.log("train_loss_step", loss, on_step=True, on_epoch=True)
        self.log("train_acc_step", accuracy, on_step=True, on_epoch=True)

    def validation_step(self, input, input_idx):
        image, label = input
        output = self.forward(image)
        loss = nn.CrossEntropyLoss()(output, label)
        accuracy = self.accuracy(output, label)
        self.log("val_loss_step", loss, on_step=True, on_epoch=True)
        self.log("val_acc_step", accuracy, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=0.0001)
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=32, shuffle=True, num_workers=2)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=32, shuffle=False, num_workers=2)