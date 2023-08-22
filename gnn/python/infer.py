#!/bin/env python
import os
import argparse
from time import time

import torch

import models
from ingress import ingress_file
from utils import GatorConfig, SimpleProgress

def infer(model, device, loader, output_csv):
    csv_rows = ["batch,edge_idx,label,score,node0_idx,node1_idx"]
    times = []
    for batch_i, data in enumerate(SimpleProgress(loader)):
        data = data.to(device)

        start = time()
        output = model(data.x, data.edge_index, data.edge_attr)
        end = time()
        times.append(end - start)

        data_to_save = (
            data.y,                 # labels
            output.squeeze(1),      # predictions
            data.edge_index[0],     # node 0 index
            data.edge_index[1]      # node 1 index
        )
        for LS_i, (truth, score, n0, n1) in enumerate(zip(*data_to_save)):
            csv_rows.append(f"{batch_i},{LS_i},{int(truth)},{float(score)},{int(n0)},{int(n1)}")

    with open(output_csv, "w") as f:
        f.write("\n".join(csv_rows))
        print(f"Wrote {output_csv}")

    return times

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Run GNN inference")
    parser.add_argument("config_json", type=str, help="config JSON")
    parser.add_argument(
        "--epoch", type=int, default=50, metavar="N",
        help="training epoch of model to use for inference (default: 50)"
    )
    parser.add_argument(
        "--root_files", type=str, default="",
        help="comma-separated list of .root files to run inferences over"
    )
    parser.add_argument(
        "--pt_files", type=str, default="",
        help="comma-separated list of .pt files to run inferences over"
    )
    args = parser.parse_args()

    config = GatorConfig.from_json(args.config_json)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    Model = getattr(models, config.model.name)
    model = Model.from_config(config)
    saved_model = config.get_outfile(subdir="models", epoch=args.epoch)
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()
    model = model.to(device)

    if "DNN" in config.model.name:
        from datasets import EdgeDataset, EdgeDataBatch
        from torch.utils.data import DataLoader

    times = []
    if args.root_files:
        for root_file in args.root_files.split(","):
            root_file_name = root_file.split("/")[-1].replace(".root", "")
            # Load sample
            loader = ingress_file(config, root_file, save=False, plot=False)
        
            if "DNN" in config.model.name:
                # Hacky PyG graph-level GNN inputs --> edge-level DNN inputs
                loader = DataLoader(
                    EdgeDataset(loader), batch_size=10000, collate_fn=lambda batch: EdgeDataBatch(batch)
                )

            # Run inferences
            print("Running inferences...")
            times += infer(
                model, device, loader, 
                config.get_outfile(subdir="inferences", tag=root_file_name, epoch=args.epoch, ext="csv")
            )
    if args.pt_files:
        for pt_file in args.pt_files.split(","):
            pt_file_name = pt_file.split("/")[-1].replace(".pt", "")
            # Load sample
            loader = torch.load(pt_file)
        
            if "DNN" in config.model.name:
                # Hacky PyG graph-level GNN inputs --> edge-level DNN inputs
                loader = DataLoader(
                    EdgeDataset(loader), batch_size=10000, collate_fn=lambda batch: EdgeDataBatch(batch)
                )

            # Run inferences
            print("Running inferences...")
            times += infer(
                model, device, loader, 
                config.get_outfile(subdir="inferences", tag=pt_file_name, epoch=args.epoch, ext="csv")
            )
    if not args.root_files and not args.pt_files:
        # Load test/train samples
        print("Loading data...")
        train_loader = torch.load(config.get_outfile(subdir="datasets", tag="train", short=True))
        test_loader = torch.load(config.get_outfile(subdir="datasets", tag="test", short=True))
        
        if "DNN" in config.model.name:
            # Hacky PyG graph-level GNN inputs --> edge-level DNN inputs
            train_loader = DataLoader(
                EdgeDataset(train_loader), batch_size=10000, collate_fn=lambda batch: EdgeDataBatch(batch)
            )
            test_loader = DataLoader(
                EdgeDataset(test_loader), batch_size=10000, collate_fn=lambda batch: EdgeDataBatch(batch)
            )

        # Run inferences
        print("Running training inferences...")
        times += infer(
            model, device, train_loader, 
            config.get_outfile(subdir="inferences", tag="train_inferences", epoch=args.epoch, ext="csv")
        )
        print("Running testing inferences...")
        times += infer(
            model, device, test_loader, 
            config.get_outfile(subdir="inferences", tag="test_inferences", epoch=args.epoch, ext="csv")
        )

    print(f"Avg. inference time: {sum(times)/len(times)}s")
    print("\nDone.\n")
