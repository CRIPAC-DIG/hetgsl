import os
import sys
import argparse
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch_geometric


from dataset_utils import build_dataset
from model import CPGNN
from util import edge_index_to_sparse_tensor


@torch.no_grad()
def test(model, data, x, adj, y, train_mask):
    model.eval()
    accs = []
    pred = F.softmax(model(x, adj, y, train_mask), dim=-1)
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        mask = mask[:, 0]
        cur_pred = pred[mask].max(1)[1]
        acc = cur_pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def main(args):
    dataset = build_dataset(args.dataset)
    data = dataset[0]
    # pdb.set_trace()
    model = CPGNN(dataset.num_features, args.hidden, int(dataset.num_classes), args)
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    adj = edge_index_to_sparse_tensor(data, edge_index)

    optimizer = torch.optim.Adam([
        dict(params=model.belief_estimator.parameters(), weight_decay=5e-4)
    ], lr=args.lr)

    train_mask = data.train_mask[:, 0]
    y = F.one_hot(data.y.long())
    for epoch in range(args.epoch):
        model.train()
        pred = model(x, adj, y, train_mask)
        # pdb.set_trace()
        loss = F.cross_entropy(pred[train_mask], data.y[train_mask].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accs = test(model, data, x, adj, y, train_mask)
        print(f'Epoch {epoch} trian_loss: {loss.item():.4f} train_acc: {accs[0]:.4f}, val_acc: {accs[1]:.4f}, test_acc: {accs[2]:.4f}')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Texas')
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--iterations', type=int, default=1)
    args = parser.parse_args()
    main(args)

