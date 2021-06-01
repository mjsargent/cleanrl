import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def KLZeroCenteredLevy(sigma_1, sigma_2):
    A = sigma_2 / (2*sigma_1)
    B = torch.log(torch.pow(sigma_1, 0.5)/2) 
    C = torch.log(torch.pow(sigma_2, 0.5)/2) 
    KL = A + B - C - 0.5 
    return KL

def levyDist(x,mu, sigma):
    A = np.power(sigma / (2*np.pi), 0.5)
    B = np.exp((-1* sigma) / 2*(x - mu))
    C = np.power(x - mu, (-3/2))
    p = A*B*C
    return p

# define net
net = nn.Sequential(nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10,5),
                    nn.ReLU(),
                    nn.Linear(5,1),
                    nn.ReLU())

opt = torch.optim.Adam(net.parameters(),lr=3e-4)
loss_fn = nn.L1Loss()
# define target parameters
sigma_low = torch.tensor([0.0001], requires_grad = False)
sigma_high = torch.tensor([0.1], requires_grad = False)
mu_low = 0
mu_high = 0

# define features 
feature_high = 2*torch.ones(10)
feature_low = torch.ones(10)

sigma_init = torch.tensor(1e-4)
mu_init = 0 

p_high = 0.5 # prob of sampling the high feature

# testing the divergnce loss
# create two continious plots:
#   one of high
#   one of low
#   look at thier progression

for i in range(10000):
    high_flag = True if np.random.rand() > p_high else False

    sigma_target = sigma_high if high_flag else sigma_low
    feature = feature_high if high_flag else feature_low
    
    sigma_out = net(feature)
    print(sigma_out)
    opt.zero_grad()
    #div = KLZeroCenteredLevy(sigma_out, sigma_target)
    
    loss = loss_fn(sigma_out, sigma_target)
    #print(loss)
    loss.backward() 
    opt.step()

x_axis = np.arange(0.1,10,0.01)
for s_tar,f in zip([sigma_low, sigma_high], [feature_low,feature_high]):
    with torch.no_grad():
        s_out = net(f)
    fig, ax = plt.subplots()
    print(f"target {s_tar}")
    print(f"actual {s_out}")
    p_target = levyDist(x_axis, 0, (s_tar.detach().numpy()))
    print(p_target[0:10]) 
    p_out = levyDist(x_axis, 0, (s_out.detach().numpy()))
    print(p_out[0:10])
    ax.plot(x_axis, p_target, 'r', p_out, 'b--')
    
    #fig.title(f"target {s_tar}")
    plt.show()

