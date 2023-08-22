#!/bin/env python
import os
import glob
import json
import random
import argparse
from time import time

import torch
import torch.optim.lr_scheduler as lr_schedulers
from torch import optim
from torch.nn import functional as F

import models
from utils import GatorConfig, print_title

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_start = time()
    n_events = len(train_loader)
    loss_sum = 0
    for event_i, data in enumerate(train_loader):
        # Log start
        if event_i % args.log_interval == 0:
            print(f"[Epoch {epoch}, {event_i}/{n_events} ({100*event_i/n_events:.2g}%)]", flush=True)
            if args.dry_run:
                break

        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze(1)

        if torch.any(torch.isnan(output)):
            raise ValueError("Output contains NaN values!")

        loss = F.binary_cross_entropy(output, y, reduction="mean")

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    print(f"train runtime: {time() - train_start:.3f}s", flush=True)
    print(f"train loss:    {loss_sum/n_events:.6f}", flush=True)
    return loss_sum/n_events

def test(model, device, test_loader):
    model.eval()
    test_start = time()
    n_events = len(test_loader)
    loss_sum = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            loss = F.binary_cross_entropy(output.squeeze(1), data.y, reduction="mean")
            loss_sum += loss.item()

    print(f"test runtime:  {time() - test_start:.3f}s", flush=True)
    print(f"test loss:     {loss_sum/n_events:.6f}", flush=True)
    return loss_sum/n_events

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Train GNN")
    parser.add_argument("config_json", type=str, help="config JSON")
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="quickly check a single pass"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, metavar="N",
        help="how many batches to wait before logging training status"
    )
    args = parser.parse_args()

    config = GatorConfig.from_json(args.config_json)

    # Write a copy of the config
    with open(config.get_outfile(ext="json", tag="config", short=False), "w") as f:
        config.dump(f)

    print_title("Configuration")
    print(config)

    print_title("Initialization")
    torch.manual_seed(config.train.seed)
    print(f"seed: {config.train.seed}")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"use_cuda: {use_cuda}")

    # Collect all graphs
    graphs = []
    for pt_file in glob.glob(f"{config.base_dir}/{config.name}/inputs/*.pt"):
        graphs += torch.load(pt_file)

    # Get test/train samples
    train_loader = None
    test_loader = None
    if config.train.get("train_frac", None):
        random.shuffle(graphs)
        n_graphs = len(graphs)
        n_train = int(n_graphs*config.train.train_frac)
        train_loader = graphs[:n_train]
        test_loader = graphs[n_train:]
    elif config.train.get("train_range", None) and config.train.get("test_range", None):
        if config.train.get("shuffle", False):
            random.shuffle(graphs)
        train_start, train_stop = config.train.train_range
        test_start, test_stop = config.train.test_range
        train_loader = graphs[train_start:train_stop]
        test_loader = graphs[test_start:test_stop]
    else:
        raise Exception("No test/train split specified")

    # Save test/train samples
    torch.save(train_loader, config.get_outfile(subdir="datasets", tag="train", short=True))
    torch.save(test_loader, config.get_outfile(subdir="datasets", tag="test", short=True))

    if "DNN" in config.model.name:
        from datasets import EdgeDataset, EdgeDataBatch
        from torch.utils.data import DataLoader
        # Hacky PyG graph-level GNN inputs --> edge-level DNN inputs
        train_loader = DataLoader(
            EdgeDataset(train_loader), batch_size=config.train.train_batch_size, shuffle=True,
            collate_fn=lambda batch: EdgeDataBatch(batch)
        )
        test_loader = DataLoader(
            EdgeDataset(test_loader), batch_size=config.train.test_batch_size, shuffle=True,
            collate_fn=lambda batch: EdgeDataBatch(batch)
        )

    # Load model
    Model = getattr(models, config.model.name)
    model = Model.from_config(config).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print(f"total trainable params: {total_trainable_params}")

    # Set up optimizer/scheduler
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.train.get("learning_rate", 0.001), 
        weight_decay=config.train.get("weight_decay", 0)
    )
    Scheduler = getattr(lr_schedulers, config.train.scheduler_name)
    scheduler = Scheduler(optimizer, **config.train.scheduler_kwargs)

    # Run loop over epochs
    history = {"train_loss": [], "test_loss": []}
    history_json = config.get_outfile(tag="history", ext="json", short=True)
    for epoch in range(1, config.train.n_epochs + 1):
        # Run testing and training
        epoch_start = time()
        print_title(f"Epoch {epoch}")
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        scheduler.step()

        print(f"total runtime: {time() - epoch_start:.3f}s", flush=True)

        # Update history JSON
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        # Save model every 5 epochs
        if not args.dry_run and (epoch % 5 == 0 or epoch == config.train.n_epochs):
            outfile = config.get_outfile(subdir="models", epoch=epoch)
            torch.save(model.state_dict(), outfile)
            print(f"Wrote {outfile}")

        # Save history JSON every 50 epochs
        if epoch % 50 == 0 or epoch == config.train.n_epochs:
            with open(history_json, "w") as f:
                json.dump(history, f)
                print(f"Wrote {history_json}")

    print("\nDone.\n")
