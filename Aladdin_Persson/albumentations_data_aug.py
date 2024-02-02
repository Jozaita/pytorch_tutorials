import os
import cv2
import albumentations as A
import numpy as np 
from PIL import Image


image = Image.open("Data_augmentation/cats_dogs_resized/cat_2.jpg")

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
    ],p=1.0)
])

images_list = [image]
image = np.array(image)
for i in range(5):
    augmentations = transforms(image=image)
    augmented_image = augmentations["image"]
    images_list.append(augmented_image)
