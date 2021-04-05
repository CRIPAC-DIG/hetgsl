import os
import sys
import argparse
import numpy as np
import pdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_utils import build_dataset, get_mask
from model import CPGNN
from util import edge_index_to_sparse_tensor, Logger, mymkdir, nowdt, set_seed, get_acc


def train(dataset, train_mask, val_mask, test_mask, args):
    model = CPGNN(dataset, args).cuda()

    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and name != 'H':
            weight_decay_params.append(param)
        if param.requires_grad and name == 'H':
            no_weight_decay_params.append(param)
    assert len(no_weight_decay_params) == 1 and len(weight_decay_params) > 0
    optimizer = torch.optim.Adam([
        dict(params=weight_decay_params, weight_decay=5e-4),
        dict(params=no_weight_decay_params, weight_decay=0.)
    ], lr=args.lr)
    x = dataset['features']
    normed_adj = dataset['normed_adj']
    raw_adj = dataset['raw_adj']
    y = dataset['labels']
    y_onehot = F.one_hot(y)

    best_val_acc = 0
    best_val_epoch = -1
    choosed_test_acc = 0

    print(f'Pre-train for {args.epoch_pretrain} epochs')
    for epoch in tqdm(range(args.epoch_pretrain)):
        model.train()
        pred = model.forward_pretrain(normed_adj, x)
        loss = F.cross_entropy(pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model.forward_pretrain(normed_adj, x)
        accs = get_acc(pred, y, train_mask, val_mask, test_mask)

        if accs[1] > best_val_acc:
            best_val_acc = accs[1]
            choosed_test_acc = accs[2]
            improved = '*'
            best_val_epoch = epoch
        else:
            improved = ''
        print(
            f'Epoch {epoch} trian_loss: {loss.item():.4f} train_acc: {accs[0]:.4f}, val_acc: {accs[1]:.4f}, test_acc: {accs[2]:.4f}/{choosed_test_acc:.4f}{improved}')

    best_val_acc = 0
    best_val_epoch = -1
    choosed_test_acc = 0

    for epoch in tqdm(range(args.epoch_pretrain, args.epoch)):
        model.train()
        if epoch == args.epoch_pretrain:
            print('\n**** Start to train LinBP ****\n')

        pred = model.forward_cpgnn(train_mask)
        loss = F.cross_entropy(pred[train_mask], y[train_mask])
        reg_h_loss = torch.norm(model.H.sum(dim=1), p=1)
        loss += reg_h_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model.forward_cpgnn(train_mask)
        accs = get_acc(pred, y, train_mask, val_mask, test_mask)

        if accs[1] > best_val_acc:
            best_val_acc = accs[1]
            choosed_test_acc = accs[2]
            improved = '*'
            best_val_epoch = epoch
        else:
            improved = ''
        print(
            f'Epoch {epoch} trian_loss: {loss.item():.4f} train_acc: {accs[0]:.4f}, val_acc: {accs[1]:.4f}, test_acc: {accs[2]:.4f}/{choosed_test_acc:.4f}{improved}')
        if epoch - best_val_epoch > args.patience:
            break
    return choosed_test_acc


def main(args):
    print(nowdt())
    set_seed(args.seed)
    dataset = build_dataset(args.dataset, to_cuda=True)

    test_accs = []
    for i, (train_mask, val_mask, test_mask) in enumerate(zip(dataset['train_masks'], dataset['val_masks'], dataset['test_masks'])):
        print(f'***** Split {i} starts *****')
        print(
            f'Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}\n')
        test_acc = train(dataset, train_mask.cuda(),
                         val_mask.cuda(), test_mask.cuda(), args)
        test_accs.append(test_acc)
        print('\n\n\n')

        print(f'For {len(test_accs)} splits')
        print(sorted(test_accs))
        print(
            f'Mean test acc {np.mean(test_accs)*100:.2f} \pm {np.std(test_accs)*100:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Texas')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epoch_pretrain', type=int, default=400)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--n_post_iter', type=int, default=1)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--graph_learn', action='store_true', default=False) #dummy

    args = parser.parse_args()
    print(args)
    main(args)
