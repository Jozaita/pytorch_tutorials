import sys
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision




#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Hyperparameters
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
n_epochs = 5

class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self,x):
        return x

model = torchvision.models.vgg16(weights="DEFAULT")
for param in model.parameters():
    param.requires_grad = False
model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512,100),nn.ReLU(),nn.Linear(100,10))
#model.classifier = nn.Linear(512,10)
model.to(device)

#Load data
train_dataset = datasets.CIFAR10(root="datasets/",
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = datasets.CIFAR10(root="datasets/",
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True)
#Initialize network
#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

#Train 
def train(train_loader,n_epochs):
    for epoch in range(n_epochs):
        print(f"Training on epoch num: {epoch}")
        for batch_idx,(data,targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)


            scores = model(data)
            loss = criterion(scores,targets)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

def check_accuracy(loader,model):
    n_correct = 0
    n_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader: 
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _,predictions = scores.max(1)
            n_correct += (predictions == y).sum()
            n_samples += predictions.size(0)

        print(f"Got {n_correct}/{n_samples} = {float(n_correct)/float(n_samples)*100:.2f} % correct")

train(train_loader,n_epochs=n_epochs)

check_accuracy(test_loader,model)