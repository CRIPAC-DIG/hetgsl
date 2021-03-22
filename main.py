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
from util import edge_index_to_sparse_tensor, Logger, mymkdir, nowdt


@torch.no_grad()
def test(model, raw_adj, normed_adj, x, y, y_onehot, train_mask, val_mask, test_mask):
    model.eval()
    accs = []
    with torch.no_grad():
        pred = F.softmax(model(raw_adj, normed_adj, x,
                               y_onehot, train_mask)[0], dim=-1)
        for mask in (train_mask, val_mask, test_mask):
            cur_pred = pred[mask].max(1)[1]
            acc = cur_pred.eq(y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs

def train(dataset, train_mask, val_mask, test_mask, args):
    # pdb.set_trace()
    model = CPGNN(dataset['num_feature'], args.hidden, dataset['num_class'], args).cuda()
    optimizer = torch.optim.Adam([
        dict(params=model.belief_estimator.parameters(), weight_decay=5e-4), 
        dict(params=model.linbp.parameters(), weight_decay=0)
    ], lr=args.lr)

    x = dataset['features'].cuda()
    normed_adj = dataset['normed_adj'].cuda()
    raw_adj = dataset['raw_adj'].cuda()
    y = dataset['labels'].cuda()
    y_onehot = F.one_hot(y)

    best_val_acc = 0
    best_val_epoch = -1
    choosed_test_acc = 0

    print(f'Pre-train for {args.epoch_one} epochs')
    for epoch in tqdm(range(args.epoch_one)):
        model.train()
        pred = model.forward_pretrain(normed_adj, x)
        loss = F.cross_entropy(pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    for epoch in tqdm(range(args.epoch_one, args.epoch)):
        if epoch == args.epoch_one:
            print('\n**** Start to train LinBP ****\n')
        model.train()
        pred, R, reg_h_loss = model(
            raw_adj, normed_adj, x, y_onehot, train_mask)
        loss = F.cross_entropy(pred[train_mask], y[train_mask]) + reg_h_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accs = test(model, raw_adj, normed_adj, x, y,
                    y_onehot, train_mask, val_mask, test_mask)
        if accs[1] > best_val_acc:
            best_val_acc = accs[1]
            choosed_test_acc = accs[2]
            improved = '*'
            best_val_epoch = epoch
        else:
            improved = ''
        print(f'Epoch {epoch} trian_loss: {loss.item():.4f} train_acc: {accs[0]:.4f}, val_acc: {accs[1]:.4f}, test_acc: {accs[2]:.4f}{improved}')
        # if epoch - best_val_epoch > 100:
        #     break
    return choosed_test_acc

def main(args):
    print(nowdt())
    dataset = build_dataset(args.dataset)

    test_accs = []
    for i, (train_mask, val_mask, test_mask) in enumerate(zip(dataset['train_masks'], dataset['val_masks'], dataset['test_masks'])):
        print(f'***** Split {i} starts *****')
        print(f'Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}\n')
        test_acc = train(dataset, train_mask.cuda(), val_mask.cuda(), test_mask.cuda(), args)
        test_accs.append(test_acc)
        # break
        print('\n\n\n')
    
    print(f'For {len(dataset["train_masks"])} splits')
    print(sorted(test_accs))
    print(f'Mean test acc {np.mean(test_accs):.4f} \pm {np.std(test_accs):.4f}')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Texas')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--epoch_one', type=int, default=400)
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--model', type=str, default='gcn')

    parser.add_argument('--graph_learn', action='store_true', default=False)
    parser.add_argument('--epsilon', type=float, default=0.)
    parser.add_argument('--num_pers', type=int, default=4)

    args = parser.parse_args()

    log_dir = 'log/cpgnn'
    mymkdir(log_dir)
    log_file_path = os.path.join(log_dir, f'{args.dataset}_cpgnn-{args.model}-{args.iterations}_log.txt')
    sys.stdout = Logger(log_file_path)

    print(args)

    main(args)

