import torch 
import torch.nn as nn

class Conv_block(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels,out_channels,**kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        return self.relu(self.batch_norm(self.conv(x)))
    
class Inception_block(nn.Module):
    def __init__(self,in_channels,out1x1,red3x3,out3x3,red5x5,out5x5,out1x1pool):
        super().__init__()

        self.branch_1 = Conv_block(in_channels,out1x1,kernel_size=1)
        self.branch_2 = nn.Sequential(
            Conv_block(in_channels,red3x3,kernel_size=1),
            Conv_block(red3x3,out3x3,kernel_size=3,padding=1),
        )
        self.branch_3 = nn.Sequential(
            Conv_block(in_channels,red5x5,kernel_size=1),
            Conv_block(red5x5,out5x5,kernel_size=5,padding=2)
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            Conv_block(in_channels,out1x1pool,kernel_size=1)
        )


    def forward(self,x):
        return torch.cat([self.branch_1(x),self.branch_2(x),self.branch_3(x),self.branch_4(x)],1)
    

class InceptionNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=1000):
        super().__init__()

        self.conv1 = Conv_block(in_channels=in_channels,
                                out_channels=64,
                                kernel_size=(7,7),
                                stride=(2,2),
                                padding=(3,3))
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv_block(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        self.inception5b = Inception_block(832,384,192,384,48,128,128)
        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024,num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc1(x)
        return x


x = torch.randn(3,3,224,224)
model = InceptionNet()
print(model(x).shape)