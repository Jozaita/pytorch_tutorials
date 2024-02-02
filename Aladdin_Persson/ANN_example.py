import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Create fully connected network
class NN(nn.Module):
    def __init__(self,input_size,n_classes) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,n_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    
model = NN(784,10)
x = torch.rand((64,784))

print(model(x).shape)

#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 1e-3
batch_size = 64
n_epochs = 1

#Load data
train_dataset = datasets.MNIST(root="dataset_MNIST/",
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_dataset = datasets.MNIST(root="dataset_MNIST/",
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True)
#Initialize network
model = NN(input_size=input_size,n_classes=num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

#Train 
def train(train_loader,n_epochs):
    for epoch in range(n_epochs):
        for batch_idx,(data,targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            data = data.reshape(data.shape[0],-1)

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
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _,predictions = scores.max(1)
            n_correct += (predictions == y).sum()
            n_samples += predictions.size(0)

        print(f"Got {n_correct}/{n_samples} = {float(n_correct)/float(n_samples)*100:.2f} % correct")

train(train_loader,n_epochs=n_epochs)

check_accuracy(test_loader,model)