#!/bin/env python

import torch
import torch_geometric
from model import InteractionNetwork

model = InteractionNetwork(hidden_size=10)
t = '(Tensor, Tensor, Tensor) -> Tensor'
model_ts = torch.jit.script(model.jittable(t))

print(model_ts)

os.system("mkdir -p /blue/p.chang/p.chang/{}/lst/GATOR/torchscript".format(os.getlogin()))

model_ts.save("/blue/p.chang/p.chang/data/lst/GATOR/torchscript/wrapped_gator.pt")
