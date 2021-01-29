import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
from .noisy_linear import NoisyLinear

def _make_layer(network_arch):
  layer = []
  for net in network_arch:
    if net=="relu":
      layer += [nn.ReLU()]
    elif net=="softmax":
      layer += [nn.Softmax(dim=-1)]
    else:
      if net.get("linear"):
        value = net.get("linear")
        layer += [nn.Linear(value[0],value[1])]
      elif net.get("noisy_linear"):
        value = net.get("noisy_linear")
        layer += [NoisyLinear(value[0],value[1],value[2])]
      elif net.get("conv"):
        value = net.get("conv")
        layer += [nn.Conv2d(value[0],value[1],value[2],value[3])]
      else:
        raise NotImplementedError("Do not support this kind of network layer")
  return nn.Sequential(*layer)

class Network(nn.Module):
  def __init__(self,network_arch):
    super(Network,self).__init__()
    self.layer = None
    self.head = None
    self.final = None
    if network_arch.get("all"):
      self.layer = _make_layer(network_arch["all"])

  def forward(self,x):
    return self.layer(x)

  def reset_noise(self):
    for layer in self.layer:
      if type(layer)==NoisyLinear:
        layer.reset_noise()

class Network2(Network):
  def __init__(self,network_arch):
    super(Network2,self).__init__(network_arch)
    if network_arch.get("head"):
      self.head = _make_layer(network_arch["head"])
    if network_arch.get("final"):
      self.final = _make_layer(network_arch["final"])

  def forward(self,x):
    x = self.head(x)
    x = self.final(x)
    return x

class DiscretePolicyNetwork(Network2):
  def forward(self,x):
    x = super().forward(x)
    x  = F.softmax(x,dim=-1)
    return x

  def sample(self,prob):
    dist = Categorical(prob)
    action = dist.sample()
    logprob = dist.log_prob(action).unsqueeze(-1)
    entropy = dist.entropy().unsqueeze(-1)
    return action,logprob,entropy  

  def get_logprob(self,prob,action):
    dist = Categorical(prob)
    logprob = dist.log_prob(action)
    return logprob

  def get_entropy(self,prob):
    dist = Categorical(prob)
    entropy = dist.entropy()
    return entropy

class ContinuousPolicyNetwork(Network2):
  def __init__(self,network_arch):
    super(ContinuousPolicyNetwork,self).__init__(network_arch)
    value = network_arch.get("final")[-1].get("linear")
    self.std_head = nn.Linear(value[0],value[1])

  def forward(self,x):
    x = self.head(x)
    mean = torch.tanh(self.final(x))*2
    std = F.softplus(self.std_head(x))+1e-3
    return mean,std

  def sample(self,prob):
    mean,std = prob
    dist = Normal(mean,std)
    action = dist.sample()
    logprob = dist.log_prob(action).sum(-1).unsqueeze(-1)
    entropy = dist.entropy().sum(-1).unsqueeze(-1)
    return action,logprob,entropy  

  def get_logprob(self,prob,action):
    mean,std = prob
    dist = Normal(mean,std)
    logprob = dist.log_prob(action).sum(-1)
    return logprob

  def get_entropy(self,prob):
    mean,std = prob
    dist = Normal(mean,std)
    entropy = dist.entropy().sum(-1)
    return entropy


class RNNNetwork(nn.Module):
  def __init__(self,network_arch):
    super(RNNNetwork,self).__init__()

  def forward(self,x,hidden):
    x_,hidden_ = self.layer(x,hidden)
    return x_,hidden_

class DummyNetwork(nn.Module):
  def __init__(self):
    super(DummyNetwork,self).__init__()

  def forward(self,x):
    return x
