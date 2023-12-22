from pytorch_lightning import Trainer
from lightning_model import VGG16Lightning
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from pytorch_lightning.loggers import CometLogger
import comet_ml

comet_ml.init(project_name="comet-vgg16-pytorch-lightning")

path = "./PetImg"
train_size = 20000
val_size = 5000

comet_logger = CometLogger()

transforms =transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((224, 224), antialias=True),
                                transforms.Normalize((0.5,),(0.5,)),
                                transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
                                transforms.RandomRotation(degrees=(0, 180)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5)])

dataset = ImageFolder(path, transform=transforms)

# print(len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, val_size])

lightning_model = VGG16Lightning(num_classes=2, train_ds=train_ds, val_ds=val_ds)

comet_logger.log_hyperparams({"batch_size": 32})

trainer = Trainer(accelerator="gpu", max_epochs=25, logger=comet_logger)
trainer.fit(model=lightning_model)

