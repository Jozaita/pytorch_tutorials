import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator,Generator,initialize_weights

device = "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)],[0.5 for _ in range(CHANNELS_IMG)])
])

dataset = datasets.MNIST(root="dataset_MNIST",train=True,transform=transforms,download=True)

loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
gen = Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG,FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)
writer_real = SummaryWriter(f"runs/DCGAN/real")
writer_fake = SummaryWriter(f"runs/DCGAN/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    print(f"EPOCH {epoch}/{NUM_EPOCHS} BEGIN")
    for batch_idx,(real,_) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(device)
        fake = gen(noise).to(device)

        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake,torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        output = disc(fake).reshape(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:16],normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:16],normalize=True)
                
                writer_fake.add_image(
                    "Mnist Fake images",img_grid_fake,global_step=step
                )

                writer_real.add_image(
                    "Mnist Real images",img_grid_real,global_step=step
                )

                step += 1
                  





