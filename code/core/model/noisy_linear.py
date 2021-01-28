import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
  def __init__(self,in_dim,out_dim,std_init):
    super(NoisyLinear,self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.std_init = std_init
    self.w_mu = nn.Parameter(torch.Tensor(self.out_dim,self.in_dim))
    self.w_sigma = nn.Parameter(torch.Tensor(self.out_dim,self.in_dim))
    self.register_buffer("w_epsilon",torch.Tensor(self.out_dim,self.in_dim))
    self.b_mu = nn.Parameter(torch.Tensor(self.out_dim))
    self.b_sigma = nn.Parameter(torch.Tensor(self.out_dim))
    self.register_buffer("b_epsilon",torch.Tensor(self.out_dim))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1/math.sqrt(self.in_dim)
    self.w_mu.data.uniform_(-mu_range,mu_range)
    self.w_sigma.data.fill_(self.std_init/math.sqrt(self.in_dim))
    self.b_mu.data.uniform_(-mu_range,mu_range)
    self.b_sigma.data.fill_(self.std_init/math.sqrt(self.out_dim))

  def reset_noise(self):
    epsilon_in = self.scale_noise(self.in_dim)
    epsilon_out = self.scale_noise(self.out_dim)
    self.w_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.b_epsilon.copy_(epsilon_out)

  def forward(self,x):
    return F.linear(x,self.w_mu+self.w_sigma*self.w_epsilon,
                    self.b_mu+self.b_sigma*self.b_epsilon)

  @staticmethod
  def scale_noise(size): # Factorized gaussian noise
    x = torch.FloatTensor(np.random.normal(loc=0.0,scale=1.0,size=size))
    return x.sign().mul(x.abs().sqrt())

