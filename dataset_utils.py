#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import math
import pdb

import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork


def build_dataset(name, split='public'):
    """
    to do: adj_remove_eye transform

    """
    root = 'data'
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset =  Planetoid(root, name, split, transform=T.NormalizeFeatures())
    if name in ['Squirrel', 'Chameleon']:
        dataset = WikipediaNetwork(root, name, transform=T.NormalizeFeatures())
    if name in ['Texas', ]:
        dataset = WebKB(root, name, transform=T.NormalizeFeatures())

    # dataset.row_normalize_features()
    # dataset.adj_remove_eye()

    return dataset


def get_mask(dataset):
    data = dataset[0]
    train_masks, val_masks, test_masks = [], [], []
    if isinstance(dataset, Planetoid):
        train_masks = [data.train_mask]
        val_masks = [data.val_mask]
        test_masks = [data.test_mask]
    
    elif isinstance(dataset, WebKB) or isinstance(dataset, WikipediaNetwork):
        for i in range(data.train_mask.shape[1]):
            train_mask = data.train_mask[:, i]
            val_mask = data.val_mask[:, i]
            test_mask = data.test_mask[:, i]

            train_masks.append(train_mask)
            val_masks.append(val_mask)
            test_masks.append(test_mask)

    return (train_masks, val_masks, test_masks)