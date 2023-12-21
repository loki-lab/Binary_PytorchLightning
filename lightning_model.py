from model import VGG16
from pytorch_lightning import LightningModule

class VGG16Lightning(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = VGG16(num_classes=num_classes)
    
    def forward(self, x):
        out = self.model(x)
        return out