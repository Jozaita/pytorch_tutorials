import os
import torch.nn as nn
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np 
from PIL import Image

class ImageFolder(nn.Module):
    def __init__(self,root_dir,transform=None):
        super(ImageFolder,self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)

        for index,name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir,name))
            self.data = zip(files,[index]*len(files))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir,self.class_names["label"])
        image = np.array(Image.open(os.path.join(root_and_dir,img_file)))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]


transforms = A.Compose([
    A.Resize(width=800,height=800),
    A.RandomCrop(width= 600,height=600),
    A.Rotate(limit=40,border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(0.5),
    A.VerticalFlip(0.1),
    A.RGBShift(25,25,25,.8),
    A.OneOf([
        A.Blur(blur_limit=3,p=0.8),
        A.ColorJitter(p=0.2)
    ],p=1.0),
    A.Normalize(mean=[0,0,0],std=[1,1,1]),
    ToTensorV2(),
],bbox_params=A.BboxParams(format="pascal_voc",label_fields=[],min_area=2048,min_visibility=0.3))


dataset = ImageFolder(root_dir="cat_dogs",transform=transforms)
