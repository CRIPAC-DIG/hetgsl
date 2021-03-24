import pdb
import sys

import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork

def edge_index_to_sparse_tensor(data, edge_index):
    n = data.x.shape[0]
    m = edge_index.shape[1]
    return torch.sparse_coo_tensor(edge_index, torch.ones(m), size=(n, n))

def normalize_sparse_tensor(adj, fill_value=1):
    """
    make row sums to 1
    """
    # pdb.set_trace()
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes= adj.size(0)
    # edge_index, edge_weight = add_self_loops(
	# edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    from torch_scatter import scatter_add
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-1.)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight

    shape = adj.shape
    return torch.sparse.FloatTensor(edge_index, values, shape)


def row_sum_one_normalize(raw_adj):
    assert raw_adj.min().item() >= 0

    row_sum = raw_adj.sum(1, keepdim=True)
    # row_sum[row_sum < 1e-5] = 1e10
    row_sum = row_sum + 1e-10
    r_inv = row_sum ** (-1)
    # r_inv[torch.isinf(r_inv)] = 0
    # r_inv = row_sum

    return raw_adj * r_inv

def set_seed(seed):
    import numpy as np
    import random
    import torch
    
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-gpu
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def nowdt():
    """
    get string representation of date and time of now()
    """
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def mymkdir(dir_name, clean=False):
    import os
    import shutil

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    elif clean:
        # rmtree(train_dir)
        # print('To remove ', train_dir)
        yn = input('Delete directory %s, y or n? ' % dir_name)
        if yn.lower() == 'y':
            shutil.rmtree(dir_name)
            print('Cleaning and recreating %s ...' % dir_name)
            os.makedirs(dir_name)


class Logger(object):
    """
    这个类的目的是尽可能不改变原始代码的情况下, 使得程序的输出同时打印在控制台和保存在文件中
    用法: 只需在程序中加入一行 `sys.stdout = Logger(log_file_path)` 即可
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
    
