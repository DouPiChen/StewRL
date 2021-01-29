import json

policy_net = {
  "head": [
    {"linear": ["in_dim",256]},
    "relu",
    {"linear": [256,256]},
    "relu",
  ],
  "final": [
    {"linear": [256,"out_dim"]},
  ]
}

value_net = {
  "all": [
    {"linear": ["in_dim",256]},
    "relu",
    {"linear": [256,256]},
    "relu",
    {"linear": [256,"out_dim"]}
  ]
}

network = dict()
network.update({"policy": policy_net})
network.update({"value": value_net})


with open("ac.json","w") as fp:
  json.dump(network,fp,indent=4)
