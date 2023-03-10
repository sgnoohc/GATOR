#!/bin/env python

import os
import argparse
from time import time
import numpy as np
import torch

import models
from configs import GatorConfig
from train import get_model_filename

def infer(model, device, test_loader):
    csv_rows = ["idx,truth,score"]
    times = []
    for event_i, data in enumerate(test_loader):
        data = data.to(device)
        model = model.to(device)

        start = time()
        output = model(data.x, data.edge_index, data.edge_attr)
        end = time()
        times.append(end - start)

        for truth, score in zip(data.y, output):
            csv_rows.append(f"{event_i},{truth},{score}")

        return csv_rows, times

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Run GNN inference")
    parser.add_argument("config_json", type=str, help="config JSON")
    parser.add_argument(
        "--epoch", type=int, default=50, metavar="N",
        help="training epoch of model to use for inference (default: 50)"
    )
    args = parser.parse_args()

    config = GatorConfig.from_json(args.config_json)
    os.makedirs(f"{config.basedir}/inference", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved_model = f"{config.basedir}/trained_models/{get_model_filename(config, args.epoch)}"
    Model = getattr(models, config.model.name)
    model = Model(config)
    model.load_state_dict(torch.load(saved_model))
    model.eval()

    test_loader = torch.load(f"{config.basedir}/{config.name}_test.pt")

    csv_rows, times = infer(model, device, test_loader)
    print(f"Avg. inference time: {sum(times)/len(times)}s")
    outfile = f"{config.basedir}/inference/{config.name}.csv"
    with open(outfile, "w") as f:
        f.write("\n".join(csv_rows))
        print(f"Wrote {outfile}")
