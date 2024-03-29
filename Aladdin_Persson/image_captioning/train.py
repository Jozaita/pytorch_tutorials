import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from custom_datasets_texts import get_loader 
from model import CNNtoRNN

def train():
    transform = transforms.Compose([
        transforms.Resize((356,356)),
        transforms.RandomCrop((299,299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    train_loader,dataset = get_loader(root_dir="flickr_8k/images",
                                      annotation_file="flickr_8k/captions.txt",
                                      transform=transform,
                                      num_workers=2)


    #Hyperparam
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 1e-4
    num_epochs = 100

    writer = SummaryWriter("runs/flickr")
    step = 0

    model = CNNtoRNN(embed_size,hidden_size,vocab_size,num_layers)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    device = "cpu"
    model.train()

    for epoch in range(num_epochs):
        for idx, (imgs,captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs,captions[:-1])
            loss = criterion(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))
            writer.add_scalar("Training loss",loss.item(),global_step=step)
            step += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    train()