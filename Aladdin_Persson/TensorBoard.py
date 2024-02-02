import sys
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter


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

#Create CNN network
class CNN(nn.Module):
    def __init__(self,in_channels,n_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1)) #Which is padding= "same"
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7,n_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.fc1(x.reshape(x.shape[0],-1))

        return x
    
model = CNN(1,10)
x = torch.rand(64,1,28,28)



#Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Hyperparameters
in_channels = 1
input_size = 784
num_classes = 10
learning_rate = 1e-3
batch_size = 64
n_epochs = 5


#Train 
def train(train_loader,n_epochs):
    step = -1

    for epoch in range(n_epochs):
        losses = []
        accs = []
        print(f"Training on epoch num: {epoch}")
        for batch_idx,(data,targets) in enumerate(train_loader):
            step += 1
            data = data.to(device)
            targets = targets.to(device)
            model.train()

            scores = model(data)
            loss = criterion(scores,targets)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            features = data.reshape(data.shape[0],-1)
            class_labels = [classes[label] for label in targets]
            img_grid = torchvision.utils.make_grid(data)


            _,predictions = scores.max(1)
            num_correct = (predictions == targets).sum()
            running_train_acc = num_correct/float(data.shape[0])
            accs.append(running_train_acc)

            writer.add_image('mnist_images',img_grid)
            writer.add_histogram('fc1',model.fc1.weight)
            writer.add_scalar('Training loss', loss,global_step=batch_idx)
            writer.add_scalar('Training acc', running_train_acc,global_step=batch_idx)
            if batch_idx == 230:
                writer.add_embedding(features,metadata=class_labels,label_img=data,global_step=batch_idx)
        writer.add_hparams({'lr':learning_rate,'bsize':batch_size},{'accuracy':np.mean(accs),'loss':np.mean(losses)})


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




batch_sizes = [256]
learning_rates = [0.01]
classes = ['0','1','2','3','4','5','6','7','8','9']

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        
        writer = SummaryWriter(f'runs/MNIST/b_s_{batch_size}_lr_{learning_rate}')
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
        #model = NN(input_size=input_size,n_classes=num_classes).to(device)
        model = CNN(in_channels=in_channels,n_classes=num_classes).to(device)
        #Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
        train(train_loader,n_epochs=n_epochs)

        check_accuracy(test_loader,model)


