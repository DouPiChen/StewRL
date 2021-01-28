import torch.nn as nn
from .network import _make_layers

class ComposedNetwork(nn.Module):
  def __init__(self,network_arch):
    super(ComposedNetwork,self).__init__()
    layers = dict()
    for key,value in network_arch.items():
      layers.update({key: _make_layers(value)})
    self.layers = layers

  def forward(self,x):
    pass

class DuelingNetwork(ComposedNetwork):
  def __init__(self,network_arch):
    super(DuelingNetwork,self).__init__(network_arch)
    self.feature_layer = self.layers["feature"]
    self.advantage_layer = self.layers["actor"]
    self.value_layer = self.layers["value"]

  def forward(self,x):
    feature = self.feature_layer(x)
    value = self.value_layer(feature)
    advantage = self.advantage_layer(feature)
    q = value + advantage-advantage.mean(dim=-1,keepdim=True)
    return q
