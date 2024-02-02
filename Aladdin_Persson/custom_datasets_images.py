import numpy as np
from skimage import io
import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class CatsAndDogsDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        print(self.annotations.shape)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = io.imread(img_path)
        if len(image.shape) == 4:
            image = np.squeeze(image,0)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))
        if self.transform:
            image = self.transform(image)
        return image,y_label
