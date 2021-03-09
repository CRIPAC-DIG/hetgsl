import pdb

import torch

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