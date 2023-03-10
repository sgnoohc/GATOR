#!/bin/env python

import os
import argparse
from time import time
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

import models
from configs import GatorConfig

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    n_batches = len(train_loader)
    for batch_i, data in enumerate(train_loader):
        # Log start of batch
        if batch_i % args.log_interval == 0:
            print(f"[Epoch {epoch}, {batch_i}/{n_batches} ({100*batch_i/n_batches:.2g}%)]")
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

    print(f"runtime: {time() - epoch_t0}s")
    print(f"train loss: {np.mean(losses)}")
    return np.mean(losses)

def validate(model, device, val_loader):
    model.eval()
    opt_thresholds, accs = [], []
    for batch_i, data in enumerate(val_loader):
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze()
        loss = F.binary_cross_entropy(output, y, reduction="mean").item()

        # define optimal threshold (thresh) where TPR = TNR
        diff, opt_thresh, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thresh in np.arange(0.001, 0.5, 0.001):
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

    print(f"val accuracy: {np.mean(accs):.4f}")
    return np.mean(opt_thresholds)

def test(model, device, test_loader, thresh=0.5):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for batch_i, data in enumerate(test_loader):
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

    print(f"test loss: {np.mean(losses):.4f}")
    print(f"test accuracy: {np.mean(accs):.4f}")
    return np.mean(losses), np.mean(accs)

if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="PyG Interaction Network Implementation")
    parser.add_argument("config_json", type=str, help="config JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="toggle verbosity")
    parser.add_argument(
        "--no_cuda", action="store_true", default=False,
        help="disables CUDA training"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=50, metavar="N",
        help="number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="quickly check a single pass"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, metavar="N",
        help="how many batches to wait before logging training status"
    )
    parser.add_argument(
        "--sample", type=int, default=1,
        help="TrackML train_{} sample to train on"
    )
    args = parser.parse_args()

    config = GatorConfig.from_json(args.config_json)

    use_cuda = torch.cuda.is_available()

    print("use_cuda={0}".format(use_cuda))

    torch.manual_seed(config.train.seed)

    print("seed={0}".format(config.train.seed))

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.load(f"{config.basedir}/{config.name}_train.pt")
    test_loader  = torch.load(f"{config.basedir}/{config.name}_test.pt")
    val_loader   = torch.load(f"{config.basedir}/{config.name}_val.pt")

    # Load model
    Model = getattr(models, config.model.name)
    model = Model(config).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print(f"total trainable params: {total_trainable_params}")

    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    scheduler = StepLR(
        optimizer, 
        step_size=config.train.learning_rate_step_size, 
        gamma=config.train.learning_rate_step_gamma
    )

    os.makedirs(f"{config.basedir}/trained_models", exist_ok=True)

    output = {"train_loss": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, args.n_epochs + 1):
        print(f"---- Epoch {epoch} ----")
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        thresh = validate(model, device, val_loader)
        print(f"optimal threshold: {thresh}")
        test_loss, test_acc = test(model, device, test_loader, thresh=thresh)
        scheduler.step()

        if not args.dry_run:
            outfile = (
                config.name
                + f"_model{config.model.name}"
                + f"_hiddensize{config.model.n_hidden_layers}"
                + f"_epoch{epoch}"
                + f"_lr{config.train.learning_rate}"
                + f"_0.8GeV_redo.pt"
            )
            torch.save(model.state_dict(), f"{config.basedir}/trained_models/{outfile}")

        output["train_loss"].append(train_loss)
        output["test_loss"].append(test_loss)
        output["test_acc"].append(test_acc)

    print(output)
