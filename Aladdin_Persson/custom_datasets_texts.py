import os
from typing import Any
import pandas as pd 
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import torchvision.transforms as transforms


spacy_eng = spacy.load("en_core_web_sm")
class Vocabulary:
    def __init__(self,freq_threshold) -> None:
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.stoi = {value:key for key,value in self.itos.items()}
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self,sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text
        ]

class FlickrDataset(Dataset):
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5) -> None:
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        #Get img,caption cols
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        #Initialize vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.to_list())
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir,img_id)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img,torch.tensor(numericalized_caption)
    
class MyCollate:
    def __init__(self,pad_idx) -> None:
        self.pad_idx = pad_idx
    
    def __call__(self, batch) -> Any:
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets,batch_first=False,padding_value=self.pad_idx)

        return imgs,targets
    
def get_loader(root_dir,annotation_file,transform,batch_size=32,num_workers=8,shuffle=True,pin_memory=True):


    dataset = FlickrDataset(root_dir,annotation_file,transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,pin_memory=pin_memory,collate_fn=MyCollate(pad_idx=pad_idx))

    return loader


transforms_ = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])
data_loader = get_loader("flickr_8k/Images/","flickr_8k/captions.txt",transform = transforms_)


