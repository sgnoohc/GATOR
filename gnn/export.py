#!/bin/env python

import torch
import torch_geometric
from model import InteractionNetwork

model = InteractionNetwork(hidden_size=10)
t = '(Tensor, Tensor, Tensor) -> Tensor'
model_ts = torch.jit.script(model.jittable(t))

print(model_ts)

model_ts.save("wrapped_gator.pt")
