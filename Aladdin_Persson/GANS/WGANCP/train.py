import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Critic,Generator,initialize_weights
from utils import gradient_penalty
device = "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)],[0.5 for _ in range(CHANNELS_IMG)])
])

dataset = datasets.MNIST(root="dataset_MNIST",train=True,transform=transforms,download=True)

loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
gen = Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN).to(device)
disc = Critic(CHANNELS_IMG,FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))
opt_disc = optim.Adam(disc.parameters(),lr=LEARNING_RATE,betas=(0.0,0.9))

fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)
writer_real = SummaryWriter(f"runs/WGANGP/real")
writer_fake = SummaryWriter(f"runs/WGANGP/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    print(f"EPOCH {epoch}/{NUM_EPOCHS} BEGIN")
    for batch_idx,(real,_) in enumerate(loader):
        real = real.to(device)

        
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(device)
            fake = gen(noise).to(device)
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)
            gp = gradient_penalty(disc,real,fake,device)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + LAMBDA_GP*gp
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step() 


        

        output = disc(fake).reshape(-1)
        loss_gen = -(torch.mean(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(f"LOSS CRIT:{loss_disc:.4f}, LOSS GEN:{loss_gen:.4f}")
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
                  





