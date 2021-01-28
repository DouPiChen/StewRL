import torch
import torch.nn as nn
from torch.distributions import Categorical,Normal
from .noisy_linear import NoisyLinear

def _make_layers(network_arch):
  layers = []
  for net in network_arch:
    if net=="relu":
      layers += [nn.ReLU()]
    elif net=="softmax":
      layers += [nn.Softmax(dim=-1)]
    else:
      if net.get("linear"):
        value = net.get("linear")
        layers += [nn.Linear(value[0],value[1])]
      elif net.get("noisy_linear"):
        value = net.get("noisy_linear")
        layers += [NoisyLinear(value[0],value[1],value[2])]
      elif net.get("conv"):
        value = net.get("conv")
        layers += [nn.Conv2d(value[0],value[1],value[2],value[3])]
      else:
        raise NotImplementedError("Do not support this kind of network layer")
  return nn.Sequential(*layers)

class Network(nn.Module):
  def __init__(self,network_arch):
    super(Network,self).__init__()
    self.layers = _make_layers(network_arch)

  def forward(self,x):
    return self.layers(x)

  def reset_noise(self):
    for layer in self.layers:
      if type(layer)==NoisyLinear:
        layer.reset_noise()

class DiscretePolicyNetwork(Network):
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

class ContinuousPolicyNetwork(Network):
  def __init__(self,network_arch):
    super(ContinuousPolicyNetwork,self).__init__(network_arch)
    network_arch_reverse = network_arch.copy()
    network_arch_reverse.reverse()
    out_dim = 1
    for layer in network_arch_reverse:
      if layer[0]=="linear":
        out_dim = layer[2]
        break
    std_init = torch.zeros(out_dim)
    self.log_stddev = nn.Parameter(std_init)

  def forward(self,x):
    mean = self.layers(x)
    std = torch.exp(self.log_stddev)
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
    x_,hidden_ = self.layers(x,hidden)
    return x_,hidden_

class DummyNetwork(nn.Module):
  def __init__(self):
    super(DummyNetwork,self).__init__()

  def forward(self,x):
    return x
