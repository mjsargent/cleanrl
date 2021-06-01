import torch
import torch.nn as nn
import numpy as np
from scipy.stats import levy
from scipy.stats import norm

def print_grad(self, grad_input, grad_output):
    print("Layer:", self.__class__.__name__)
    print("grad_input:", grad_input)
    print("grad_input_norm:", grad_input[0].norm())
    print("grad_output:", grad_output)
    print("grad_output_norm:", grad_output[0].norm())

np.random.seed(0)


loss = nn.MSELoss()
target_set = torch.tensor([[i**2] for i in range(10)], dtype=torch.float)
print(target_set)

x_set = torch.tensor([[i]*10 for i in range(10)], dtype=torch.float)
print(x_set)
net = nn.Sequential(
    nn.Linear(10, 2),
    nn.ReLU()
)
class TestNet(nn.Module):
    def __init__(self, input_size, output_size, use_levy=True):
        super(TestNet,self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc_mu = nn.Linear(input_size,1)
        self.fc_scale = nn.Linear(input_size,1 )
        nn.init.uniform_(self.fc_mu.weight)
        nn.init.uniform_(self.fc_scale.weight)
        self.activation = nn.ReLU()
        self.use_levy = use_levy

    def forward(self, x):
       mu = self.activation(self.fc_mu(x))
       scale = self.activation(self.fc_scale(x)) 
       if self.use_levy:
           n = mu + scale * torch.tensor((norm.ppf(1 - np.random.rand(1))**-2), dtype=torch.float)
       else:
           n = mu + scale
       return n, (mu, scale)

net = TestNet(10,2,True)
net.fc_mu.register_backward_hook(print_grad)
net.fc_scale.register_backward_hook(print_grad)
opt = torch.optim.SGD(net.parameters(), lr=0.1)

epochs = 100
# test diff through reparam dist

print("Reparm")
use_levy = True
for i in range(epochs):
    for x,target in zip(x_set, target_set):
        n,z = net(x)
        error = loss(n, target)
        print(f"Levy Parameters: mu:{z[0]} scale{z[1]}")
        print(f"Output:{n} | loss:{error}")

        opt.zero_grad()
        error.backward()
        opt.step()

# test differentiing through levy dist 
for i in range(epochs):
    z = net.forward(x)
    n = levy.rvs(loc=z[0], scale=z[1], size=1)    
     
    error = loss(m, target)
    print(f"Output:{n} | loss:{error}")
    opt.zero_grad() 
    error.backward()
    opt.step()
