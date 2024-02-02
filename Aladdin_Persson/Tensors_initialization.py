import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,
                         device=device,requires_grad=True)

print(my_tensor)

#Other initialization methods 

x = torch.empty(size=(3,3))
y = torch.zeros((3,3))
z = torch.rand((3,3))
a = torch.ones((3,3))
b = torch.eye(5)
c = torch.arange(0,5,1)
d = torch.linspace(0.1,10,10)
e = torch.empty((1,5)).normal_(mean=0,std=1)
f = torch.diag(torch.ones(3))

#print(f)

#How to initialize and convert tensors to other dtypes
tensor = torch.arange(4)

#print(tensor.bool())
#print(tensor.short())
#print(tensor.long())
#print(tensor.half())
#print(tensor.float())
#print(tensor.double())

# Numpy to tensor 
import numpy as np

np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np.array = tensor.numpy()

#tensor math and operations 

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

#Addition 
z1 = torch.empty(3)
torch.add(x,y,out=z1)

z2 = torch.add(x,y)

z3 = x+y

#Substraction

z1 = torch.empty(3)
torch.subtract(x,y,out=z1)

z2 = torch.subtract(x,y)

z3 = x-y

#Division
z = torch.true_divide(x,y)
#Inplace operations,followed by _
t = torch.zeros(3)
t.add_(x)
t+=x #but not t = t+x

#Exponentiation

z = x.pow(2)
z = x**2

#Comparisons

z = x>0
print(z)

#Matrix multiplication 

a_1 = torch.rand((2,5))
a_2 = torch.rand((5,3))
a_3 = a_1.mm(a_2)
x_3 = torch.mm(a_1,a_2)

#Matrix exponentiation 

matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)

#Element wise multiplcation
z = x*y
print(z)

#dot product

print(torch.dot(x,y))

#Batch matrix multiplication
batch = 32
n = 10
m = 30
p= 20

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmn = torch.bmm(tensor1,tensor2)

#Broadcasting,expands arrays to perform the operations, 
# Add the dimensiones required
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1-x2
z_2 = x1**x2

print(x1)
print(x2)
print(z)

#Other operations 

sum_x = torch.sum(x,dim=0)
values,indices = torch.max(x,dim=0)
values,indices = torch.min(x,dim=0)

z = torch.abs(x)
z_2 = torch.argmax(x,dim=0)
z_3 = torch.argmin(x,dim=0)
mean_x = torch.mean(x.float(),dim=0)
z = torch.eq(x,y)
sorted_y, indices = torch.sort(y,dim=0,descending=False)
z = torch.clamp(x,min=0) #Set a min value in the array


x = torch.tensor([1,0,0,1,1,0],dtype=torch.bool)
z = torch.any(x)
z_2 = torch.all(x)
print(z)
print(z_2)
#Most of these functions are also methods of a tensor object

#Tensor indexing
batch_size = 10 
features = 25

x = torch.rand((batch_size,features))

print(x[0].shape)
print(x[:,0].shape)
print(x[2,:10])

#Fancy indexing

x = torch.arange(10)
indices = [2,5,8]
print(x[indices])
x = torch.rand((3,3))

rows = torch.tensor([1,0])
columns = torch.tensor([1,2])
print(x[rows,columns].shape)

#More advanced indexing
x = torch.arange(10)
print(x[(x<2)|(x>8)])
print(x[x.remainder(2) ==0])
print(torch.where(x>5,x,x*2))

print(torch.tensor([0,0,0,0,0,1,1,1,1,2,2]).unique())

print(x.ndimension())
print(x.numel())

#Reshaping
x = torch.arange(9)
#Difference between view and reshape: view uses the contiguous
#spaces for matrix in memory. If the matrix is modified internally
#probably the contiguous spaces may move, and .contiguous() is necessary
x_1 = x.view(3,3)
x_2 = x.reshape(3,3)


print(torch.cat((x_1,x_2),dim=0))
#flatten 
z = x_1.view(-1)

batch = 64
x = torch.rand((batch,2,5))
z = x.view(batch,-1)
print(z.shape)
#Switch axis
z = x.permute(0,2,1)
print(z.shape)

#Transpose for more dimensions

x = torch.arange(10)
print(x.shape)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)
#Reverse with squeeze