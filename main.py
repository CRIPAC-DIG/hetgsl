import os
import sys
import argparse
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch_geometric
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork


from dataset_utils import build_dataset, get_mask
from model import CPGNN
from util import edge_index_to_sparse_tensor, Logger, mymkdir, nowdt


@torch.no_grad()
def test(model, data, x, adj, y, train_mask, val_mask, test_mask):
    model.eval()
    accs = []
    with torch.no_grad():
        pred = F.softmax(model(x, adj, y, train_mask)[0], dim=-1)
        # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        #     mask = mask[:, 0]
        for mask in (train_mask, val_mask, test_mask):
            cur_pred = pred[mask].max(1)[1]
            acc = cur_pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs

def train(dataset, train_mask, val_mask, test_mask, args):
    model = CPGNN(dataset.num_features, args.hidden, int(dataset.num_classes), args)
    optimizer = torch.optim.Adam([
        dict(params=model.belief_estimator.parameters(), weight_decay=5e-4), 
        dict(params=model.linbp.parameters(), weight_decay=0)
    ], lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    data = dataset[0]
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    adj = edge_index_to_sparse_tensor(data, edge_index)

    y = F.one_hot(data.y.long())

    best_val_acc = 0
    choosed_test_acc = 0

    for epoch in range(args.epoch_one):
        model.train()
        pred = model.forward_one(x, adj, y, train_mask)
        # pdb.set_trace()
        loss = F.cross_entropy(pred[train_mask], data.y[train_mask].long())
        # loss += 0.0005 * (torch.norm(model.belief_estimator.lin1.weight) ** 2 + torch.norm(model.belief_estimator.lin2.weight) ** 2 )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accs = test(model, data, x, adj, y, train_mask, val_mask, test_mask)
        print(f'Epoch {epoch} trian_loss: {loss.item():.4f} train_acc: {accs[0]:.4f}, val_acc: {accs[1]:.4f}, test_acc: {accs[2]:.4f}')
    
    for epoch in range(args.epoch_one, args.epoch):
        if epoch == args.epoch_one:
            print('\n**** Start to train LinBP ****\n')
        model.train()
        pred, R, reg_h_loss = model(x, adj, y, train_mask)
        # pdb.set_trace()
        # loss = F.cross_entropy(pred[train_mask], data.y[train_mask].long()) + F.cross_entropy(R[train_mask], data.y[train_mask].long()) + reg_h_loss
        loss = F.cross_entropy(pred[train_mask], data.y[train_mask].long()) + reg_h_loss
        # loss += 0.0005 * (torch.norm(model.belief_estimator.lin1.weight) ** 2 + torch.norm(model.belief_estimator.lin2.weight) ** 2 )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accs = test(model, data, x, adj, y, train_mask, val_mask, test_mask)
        if accs[1] > best_val_acc:
            best_val_acc = accs[2]
            choosed_test_acc = accs[2]
            improved = '*'
        else:
            improved = ''
        print(f'Epoch {epoch} trian_loss: {loss.item():.4f} train_acc: {accs[0]:.4f}, val_acc: {accs[1]:.4f}, test_acc: {accs[2]:.4f}{improved}')
    return choosed_test_acc

def main(args):
    print(nowdt())
    dataset = build_dataset(args.dataset)
    train_masks, val_masks, test_masks = get_mask(dataset)

    test_accs = []
    for i, (train_mask, val_mask, test_mask) in enumerate(zip(train_masks, val_masks, test_masks)):
        print(f'***** Split {i} starts *****')
        print(f'Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}\n')
        test_acc = train(dataset, train_mask, val_mask, test_mask, args)
        test_accs.append(test_acc)
        # break
        print('\n\n\n')
    
    print(f'Mean test acc {np.mean(test_accs):.4f} \pm {np.std(test_accs):.4f}')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Texas')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epoch_one', type=int, default=400)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--iterations', type=int, default=2)
    args = parser.parse_args()

    log_dir = 'log/cpgnn'
    mymkdir(log_dir)
    log_file_path = os.path.join(log_dir, f'{args.dataset}_cpgnn{args.iterations}_log.txt')
    sys.stdout = Logger(log_file_path)

    print(args)

    main(args)

