import torch.nn as nn
from .network import _make_layer

class ComposedNetwork(nn.Module):
  def __init__(self,network_arch):
    super(ComposedNetwork,self).__init__()
    layer = dict()
    for key,value in network_arch.items():
      layer.update({key: _make_layer(value)})
    self.layer = layer

  def forward(self,x):
    pass

class DuelingNetwork(ComposedNetwork):
  def __init__(self,network_arch):
    super(DuelingNetwork,self).__init__(network_arch)
    self.feature_layer = self.layer["feature"]
    self.advantage_layer = self.layer["actor"]
    self.value_layer = self.layer["value"]

  def forward(self,x):
    feature = self.feature_layer(x)
    value = self.value_layer(feature)
    advantage = self.advantage_layer(feature)
    q = value + advantage-advantage.mean(dim=-1,keepdim=True)
    return q
