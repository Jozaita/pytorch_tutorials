import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_fn(disc,gen,loader,opt_disc,opt_gen,l2,bce):
    loop = tqdm(loader,leave=True)

    for idx,(x,y) in enumerate(loop):
        x,y = x.to(config.DEVICE),y.to(config.DEVICE)


        y_fake = gen(x)
        D_real = disc(x,y)
        D_fake = disc(x,y_fake.detach())
        D_real_loss = bce(D_real,torch.ones_like(D_real))
        D_fake_loss = bce(D_fake,torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss)/2

        disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        D_fake = disc(x,y_fake)
        G_fake_loss = bce(D_fake,torch.ones_like(D_fake))
        L1 = l1(y_fake,y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_dataset = MapDataset(root_dir="data/maps/train")
    train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS)
    val_dataset = MapDataset(root_dir="data/maps/val")
    val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE)

if __name__ == "__main__":
    main()