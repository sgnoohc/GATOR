#!/bin/env python
import os
import json
import argparse
from time import time
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_schedulers
from torch import optim
from torch.nn import functional as F

import models
from utils import GatorConfig, print_title

def get_model_filename(config, epoch):
    return (
        config.name
        + f"_model{config.model.name}"
        + f"_nhidden{config.model.n_hidden_layers}"
        + f"_hiddensize{config.model.hidden_size}"
        + f"_epoch{epoch}"
        + f"_lr{config.train.scheduler_name}{config.train.learning_rate}"
        + f"_0.8GeV_model.pt"
    )

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    n_events = len(train_loader)
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

        if args.verbose:
            print("target vs. output:")
            for ii, (yy, oo) in enumerate(zip(y, output)):
                if ii < 0:
                    print(yy, oo)

        if torch.any(torch.isnan(output)):
            raise ValueError("Output contains NaN values!")

        loss = F.binary_cross_entropy(output, y, reduction="mean")

        if args.verbose:
            print("before backward propagation:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

        loss.backward()
        optimizer.step()

        if args.verbose:
            print("after backward propagation:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)

        losses.append(loss.item())

    print(f"train runtime: {time() - epoch_t0}s", flush=True)
    print(f"train loss:    {np.mean(losses):0.6f}", flush=True)
    return np.mean(losses)

def validate(model, device, val_loader):
    model.eval()
    opt_thresholds, accs = [], []
    for data in val_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze(1)
        loss = F.binary_cross_entropy(output, y, reduction="mean").item()

        # define optimal threshold (thresh) where TPR = TNR
        diff, opt_thresh, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thresh in np.linspace(0.001, 0.999, 500):
            TP = torch.sum((y == 1) & (output >= thresh)).item()
            TN = torch.sum((y == 0) & (output <  thresh)).item()
            FP = torch.sum((y == 0) & (output >= thresh)).item()
            FN = torch.sum((y == 1) & (output <  thresh)).item()
            acc = (TP+TN)/(TP+TN+FP+FN)
            TPR, TNR = TP/(TP+FN), TN/(TN+FP)
            delta = abs(TPR-TNR)
            if (delta < diff):
                diff, opt_thresh, opt_acc = delta, thresh, acc

        opt_thresholds.append(opt_thresh)
        accs.append(opt_acc)

    print(f"val accuracy:  {np.mean(accs):0.6f}", flush=True)
    return np.mean(opt_thresholds)

def test(model, device, test_loader, thresh=0.5):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            TP = torch.sum((data.y == 1).squeeze() & (output >= thresh).squeeze()).item()
            TN = torch.sum((data.y == 0).squeeze() & (output <  thresh).squeeze()).item()
            FP = torch.sum((data.y == 0).squeeze() & (output >= thresh).squeeze()).item()
            FN = torch.sum((data.y == 1).squeeze() & (output <  thresh).squeeze()).item()
            acc = (TP+TN)/(TP+TN+FP+FN)
            loss = F.binary_cross_entropy(output.squeeze(1), data.y, reduction="mean").item()
            accs.append(acc)
            losses.append(loss)

    print(f"test loss:     {np.mean(losses):0.6f}", flush=True)
    print(f"test accuracy: {np.mean(accs):0.6f}", flush=True)
    return np.mean(losses), np.mean(accs)

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Train GNN")
    parser.add_argument("config_json", type=str, help="config JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="toggle verbosity")
    parser.add_argument(
        "--no_cuda", action="store_true", default=False,
        help="disables CUDA training"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=50, metavar="N",
        help="number of epochs to train"
    )
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
    os.makedirs(f"{config.basedir}/{config.name}/models", exist_ok=True)

    print_title("Configuration")
    print(config)

    print_title("Initialization")
    torch.manual_seed(config.train.seed)
    print(f"seed: {config.train.seed}")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"use_cuda: {use_cuda}")

    train_loader = torch.load(f"{config.basedir}/{config.name}/datasets/{config.name}_train.pt")
    test_loader  = torch.load(f"{config.basedir}/{config.name}/datasets/{config.name}_test.pt")
    val_loader   = torch.load(f"{config.basedir}/{config.name}/datasets/{config.name}_val.pt")

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
        val_loader = DataLoader(
            EdgeDataset(val_loader), batch_size=config.train.val_batch_size, shuffle=True,
            collate_fn=lambda batch: EdgeDataBatch(batch)
        )

    # Load model
    Model = getattr(models, config.model.name)
    model = Model(config).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print(f"total trainable params: {total_trainable_params}")

    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.train.get("learning_rate", 0.001), 
        weight_decay=config.train.get("weight_decay", 0)
    )
    Scheduler = getattr(lr_schedulers, config.train.scheduler_name)
    scheduler = Scheduler(optimizer, **config.train.scheduler_kwargs)

    history = {"train_loss": [], "test_loss": [], "test_acc": []}
    history_json = f"{config.basedir}/{config.name}/{config.name}_history.json"
    for epoch in range(1, args.n_epochs + 1):
        epoch_t0 = time()
        print_title(f"Epoch {epoch}")
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        thresh = validate(model, device, val_loader)
        print(f"best thresh:   {thresh}", flush=True)
        test_loss, test_acc = test(model, device, test_loader, thresh=thresh)
        scheduler.step()

        print(f"total runtime: {time() - epoch_t0:0.3f}s", flush=True)

        if not args.dry_run and epoch % 5 == 0:
            outname = f"{config.basedir}/{config.name}/models/{get_model_filename(config, epoch)}"
            torch.save(model.state_dict(), outname)
            print(f"Wrote {outname}")

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        if epoch % 100 == 0 or epoch == args.n_epochs:
            with open(history_json, "w") as f:
                json.dump(history, f)
                print(f"Wrote {history_json}")
