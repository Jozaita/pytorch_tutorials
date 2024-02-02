import os
import cv2
import albumentations as A
import numpy as np 
from PIL import Image


image = cv2.imread("Data_augmentation/cats_dogs_resized/cat_2.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
bboxes = [[10,20,20,10]] # Where the target is

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
],bbox_params=A.BboxParams(format="pascal_voc",label_fields=[],min_area=2048,min_visibility=0.3))

images_list = [image]
saved_bboxes = [bboxes[0]]
for i in range(5):
    augmentations = transforms(image=image,bboxes=bboxes)
    augmented_image = augmentations["image"]
    if len(augmentations["bboxes"]):
        images_list.append(augmented_image)
        saved_bboxes.append(augmentations["bboxes"][0])
