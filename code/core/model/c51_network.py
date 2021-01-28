import torch
import torch.nn as nn
import torch.nn.functional as F
from .network import _make_layers

class C51Network(nn.Module):
  def __init__(self,network_arch,infos):
    super(C51Network,self).__init__()
    self.atom_size = infos["atom_size"]
    self.support = infos["support"]
    self.layers = _make_layers(network_arch)

  def forward(self,x):
    dist = self.dist(x)
    q = torch.sum(dist*self.support,dim=2)
    return q.squeeze(0)

  def dist(self,x):
    q_atom = self.layers(x)
    out_dim = q_atom.size()[-1]/self.atom_size
    q_atom = q_atom.view(-1,int(out_dim),self.atom_size)
    distribution = F.softmax(q_atom,dim=-1)
    distribution = distribution.clamp(min=1e-3)
    return distribution

