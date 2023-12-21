from pytorch_lightning import Trainer
from lightning_model import VGG16Lightning
from torchvision import transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

path = "./PetImg"
train_size = 20000
val_size = 5000

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

trainer = Trainer(accelerator="gpu", max_epochs=25)
trainer.fit(model=lightning_model)

