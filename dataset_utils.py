#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import math
import pdb
import os
import scipy.sparse as sp

import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork


def to_scipy_sparse_matrix(edge_index, edge_attr, num_node):
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    row, col = edge_index

    if edge_attr is None:
        edge_attr = np.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1).cpu().numpy()
        assert edge_attr.size(0) == row.size(0)
    N = num_node

    out = sp.csr_matrix((edge_attr, (row, col)), (N, N))
    return out


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_sparse_adj(mx):
    """Row-normalize sparse matrix: symmetric normalized Laplacian"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def build_dataset(name, sparse_init_adj=False, to_cuda=False):
    """
    to do: adj_remove_eye transform

    """
    root = 'data'
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset =  Planetoid(root, name, 'random', transform=T.NormalizeFeatures())
    elif name in ['Squirrel', 'Chameleon']:
        dataset = WikipediaNetwork(root, name, transform=T.NormalizeFeatures())
    elif name in ['Texas','Wisconsin', 'Cornell' ]:
        dataset = WebKB(root, name, transform=T.NormalizeFeatures())
    else:
        raise NotImplementedError(f'Not implemented for dataset {name}')

    data = dataset[0]
    features = data.x 
    labels = data.y.long()
    raw_adj = to_scipy_sparse_matrix(data.edge_index, data.edge_attr, len(data.y))

    print(f'For dataset {name}')
    data = dataset[0]
    print(f'{len(data.y)} nodes, {data.edge_index.shape[-1]} edges')
    # return dataset
    train_masks, val_masks, test_masks = get_mask(dataset)

    adj = raw_adj + sp.eye(raw_adj.shape[0])
    normed_adj = normalize_sparse_adj(adj)


    if sparse_init_adj:
        raw_adj = sparse_mx_to_torch_sparse_tensor(raw_adj)
        normed_adj = sparse_mx_to_torch_sparse_tensor(normed_adj)
    else:
        # pdb.set_trace()
        raw_adj = torch.Tensor(raw_adj.todense())
        normed_adj = torch.Tensor(normed_adj.todense())
    
    num_feature = features.shape[1]
    num_class = labels.max().item() + 1

    dataset = {
        'raw_adj': raw_adj,
        'normed_adj': normed_adj,
        'features': features,
        'labels': labels,
        'train_masks': train_masks,
        'val_masks': val_masks,
        'test_masks': test_masks,
        'num_feature': num_feature,
        'num_class': num_class
    }
    if to_cuda:
        for key in dataset:
            if isinstance(dataset[key], torch.Tensor):
                dataset[key] = dataset[key].cuda()
    return dataset

def get_mask(dataset):
    data = dataset[0]
    train_masks, val_masks, test_masks = [], [], []
    if isinstance(dataset, Planetoid):
        raw_dir = dataset.raw_dir
        splits = os.listdir(raw_dir)
        splits = [os.path.join(raw_dir, split) for split in splits if split.endswith('.npz')]
        assert len(splits) == 10

        for f in splits:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]

        return (train_masks, val_masks, test_masks)

    
    elif isinstance(dataset, WebKB) or isinstance(dataset, WikipediaNetwork):
        for i in range(data.train_mask.shape[1]):
            train_mask = data.train_mask[:, i]
            val_mask = data.val_mask[:, i]
            test_mask = data.test_mask[:, i]

            train_masks.append(train_mask)
            val_masks.append(val_mask)
            test_masks.append(test_mask)

    return (train_masks, val_masks, test_masks)
