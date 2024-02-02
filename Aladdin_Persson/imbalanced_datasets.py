import torch 
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler,DataLoader
import torchvision.transforms as transforms
import torch.nn as nn


#Class weighting
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1,50]))


def get_loader(root_dir,batch_size):
    my_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=root_dir,transform=my_transforms)
    
    class_weights = []
    
    for root,subdir,files in os.walk(root_dir):
        if len(files)>0:
            class_weights.append(1/len(files))

    sample_weight = [0]*len(dataset)

    for idx,(data,label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weight[idx] = class_weight
    
    sampler = WeightedRandomSampler(sample_weight,num_samples=len(sample_weight),replacement=True)

    loader = DataLoader(dataset,batch_size=batch_size,sampler=sampler)

    return loader



def main():
    data_loader = get_loader('dataset',batch_size=8)

if __name__ == "__main__":
    main()