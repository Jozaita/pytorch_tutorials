import torch 
from torchvision.transforms import transforms
from torchvision.utils import save_image
from custom_datasets_images import CatsAndDogsDataset


my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomGrayscale(0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0,0.0,0.0],std=[1,1,1])
])

dataset = CatsAndDogsDataset(csv_file="Data_augmentation/cats_dogs.csv",root_dir="Data_augmentation/cats_dogs_resized",transform=my_transforms)

img_num = 0
for _ in range(2):
    for img,label in dataset:
        save_image(img,f'img_{img_num}.png')
        img_num += 1