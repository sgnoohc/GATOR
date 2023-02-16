#!/bin/env python

import os
import argparse
from time import time

import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn.functional as F

from model import InteractionNetwork

verbose = False

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze(1)

        # print("...target vs. output")
        # for ii, (yy, oo) in enumerate(zip(y, output)):
        #     if ii < 0:
        #         print(yy, oo)

        loss = F.binary_cross_entropy(output, y, reduction='mean')

        # print("...before backward propagation")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        loss.backward()
        optimizer.step()

        # print("...after backward propagation")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        losses.append(loss.item())
    print("...epoch time: {0}s".format(time()-epoch_t0))
    print("...epoch {}: train loss={}".format(epoch, np.mean(losses)))
    return np.mean(losses)

def validate(model, device, val_loader):
    model.eval()
    opt_thlds, accs = [], []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze()
        loss = F.binary_cross_entropy(output, y, reduction='mean').item()

        # print(output)

        # define optimal threshold (thld) where TPR = TNR
        diff, opt_thld, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thld in np.arange(0.001, 0.5, 0.001):
            TP = torch.sum((y==1) & (output>thld)).item()
            TN = torch.sum((y==0) & (output<thld)).item()
            FP = torch.sum((y==0) & (output>thld)).item()
            FN = torch.sum((y==1) & (output<thld)).item()
            acc = (TP+TN)/(TP+TN+FP+FN)
            TPR, TNR = TP/(TP+FN), TN/(TN+FP)
            delta = abs(TPR-TNR)
            if (delta < diff):
                diff, opt_thld, opt_acc = delta, thld, acc

        opt_thlds.append(opt_thld)
        accs.append(opt_acc)

    # print(accs)
    # print(opt_thlds)

    print("...val accuracy=", np.mean(accs))
    return np.mean(opt_thlds)

def test(model, device, test_loader, thld=0.5):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            TP = torch.sum((data.y==1).squeeze() &
                           (output>thld).squeeze()).item()
            TN = torch.sum((data.y==0).squeeze() &
                           (output<thld).squeeze()).item()
            FP = torch.sum((data.y==0).squeeze() &
                           (output>thld).squeeze()).item()
            FN = torch.sum((data.y==1).squeeze() &
                           (output<thld).squeeze()).item()
            acc = (TP+TN)/(TP+TN+FP+FN)
            loss = F.binary_cross_entropy(output.squeeze(1), data.y,
                                          reduction='mean').item()
            accs.append(acc)
            losses.append(loss)
            #print(f"acc={TP+TN}/{TP+TN+FP+FN}={acc}")

    print('...test loss: {:.4f}\n...test accuracy: {:.4f}'
          .format(np.mean(losses), np.mean(accs)))
    return np.mean(losses), np.mean(accs)

def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--step-size', type=int, default=5,
                        help='Learning rate step size')
    parser.add_argument('--pt', type=str, default='2',
                        help='Cutoff pt value in GeV (default: 2)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--construction', type=str, default='heptrkx_classic',
                        help='graph construction method')
    parser.add_argument('--sample', type=int, default=1,
                        help='TrackML train_{} sample to train on')
    parser.add_argument('--hidden-size', type=int, default=200,
                        help='Number of hidden units per layer')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    print("use_cuda={0}".format(use_cuda))

    torch.manual_seed(args.seed)

    print("seed={0}".format(args.seed))

    device = torch.device("cuda" if use_cuda else "cpu")

     # train_loader = torch.load("/blue/p.chang/p.chang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnGraph_ttbar_PU200_train.pt")
     # test_loader  = torch.load("/blue/p.chang/p.chang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnGraph_ttbar_PU200_test.pt")
     # val_loader   = torch.load("/blue/p.chang/p.chang/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnGraph_ttbar_PU200_valid.pt")

    train_loader = torch.load("/blue/p.chang/{}/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnUndirGraph_ttbar_PU200_train.pt".format(os.getlogin()))
    test_loader  = torch.load("/blue/p.chang/{}/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnUndirGraph_ttbar_PU200_test.pt".format(os.getlogin()))
    val_loader   = torch.load("/blue/p.chang/{}/data/lst/GATOR/CMSSW_12_2_0_pre2/LSTGnnUndirGraph_ttbar_PU200_valid.pt".format(os.getlogin()))

    model = InteractionNetwork(args.hidden_size).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print('total trainable params:', total_trainable_params)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size,
                       gamma=args.gamma)

    os.system("mkdir -p /blue/p.chang/{}/data/lst/GATOR/trained_models".format(os.getlogin()))

    output = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(1, args.epochs + 1):
        print("---- Epoch {} ----".format(epoch))
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        thld = validate(model, device, val_loader)
        print('...optimal threshold', thld)
        test_loss, test_acc = test(model, device, test_loader, thld=thld)
        scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(),
                       #"/blue/p.chang/p.chang/data/lst/GATOR/trained_models/train_hiddensize{}_PyG_LST_epoch{}_lr{}_0.8GeV_redo.pt"
                       "/blue/p.chang/{}/data/lst/GATOR/trained_models/train_hiddensize{}_PyG_LST_epoch{}_lr{}_0.8GeV_redo.pt".format(os.getlogin())
                       .format(os.getlogin(), args.hidden_size, epoch, args.lr))

        output['train_loss'].append(train_loss)
        output['test_loss'].append(test_loss)
        output['test_acc'].append(test_acc)

        # np.save('train_output/train{}_PyG_{}_{}GeV_redo'
        #         .format(args.sample, args.construction, args.pt),
        #         output)

    print(output)

if __name__ == "__main__":

    main()

