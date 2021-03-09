#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import math

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