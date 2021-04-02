import pdb
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import normalize_sparse_tensor, row_sum_one_normalize


class MLP(nn.Module):
    def __init__(self, num_features, hidden, out_dim, dropout):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(num_features, hidden)
        # pdb.set_trace()
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, input, edge_index, edge_weight):
        x = self.lin1(input)
        x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return self.lin2(x)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            # self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
            self.bias = nn.Parameter(nn.init.zeros_(self.bias))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, normed_adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(normed_adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.conv1 = GCNLayer(nfeat, nhid)
        self.conv2 = GCNLayer(nhid, nclass)

    def forward(self, normed_adj, features):
        node_vec = torch.relu(self.conv1(features, normed_adj))
        node_vec = F.dropout(node_vec, self.dropout, training=self.training)

        logits = self.conv2(node_vec, normed_adj)

        # return logits, node_vec
        return logits, node_vec

class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.
    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """

    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(
            K, in_c, out_c))  # [K, in_c, out_c]
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K

    def forward(self, inputs, laplacian, lambda_max=2):
        """
        :param inputs: the input data, [B, N, D]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, C]
        """
        # pdb.set_trace()
        I = torch.eye(laplacian.shape[0]).cuda()
        # L = I - normalize_adj(graph)
        # L = 2 * laplacian / lambda_max - I
        L = laplacian

        Tx_0 = inputs
        Tx_1 = inputs  # dummy
        out = torch.matmul(Tx_0, self.weight[0])

        if self.K > 1:
            Tx_1 = L @ inputs
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.K):
            Tx_2 = 2. * L @ Tx_1 - Tx_0
            out = out + Tx_2 @ self.weight[k]
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out


class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout, K=2):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        :param K:
        """
        super(ChebNet, self).__init__()
        self.dropout = dropout
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)

    def forward(self, laplacian, input):
        node_vec = F.dropout(F.relu(self.conv1(input, laplacian)),
                             p=self.dropout, training=self.training)
        logits = self.conv2(node_vec, laplacian)
        return logits.squeeze(), node_vec.squeeze()

class CompatibilityLayer(nn.Module):
    @staticmethod
    def makeDoubleStochasticH(H, max_iterations=float('inf'), delta=1e-7):
        converge = False
        prev_H = H
        i = 0
        while not converge and i < max_iterations:
            prev_H = H
            # H /= tf.reduce_sum(H, axis=0, keepdims=True)
            H /= H.sum(dim=0, keepdims=True)
            # H /= tf.reduce_sum(H, axis=1, keepdims=True)
            H /= H.sum(dim=1, keepdims=True)

            # delta = tf.linalg.norm(H - prev_H, ord=1)
            delta = torch.norm(H - prev_H, p=1)
            if delta < 1e-12:
                converge = True
            i += 1
        if i == max_iterations:
            warnings.warn(
                "makeDoubleStochasticH: maximum number of iterations reached.")
        return H

    @staticmethod
    def makeSymmetricH(H):
        return 0.5 * (H + H.T)

    @classmethod
    def estimateH(self, raw_adj, y_onehot, logits=None, sample_mask=None):
        """
        logits: logits
        """
        # raw_normed_adj = normalize_sparse_tensor(raw_adj)  # make row sum to 1
        row_normed_adj = row_sum_one_normalize(raw_adj)

        inputs = F.softmax(logits, dim=1)
        inputs = inputs * (1 - sample_mask[:, None].float()) + y_onehot * sample_mask[:, None]  # eq 10
        y_onehot = y_onehot * sample_mask[:, None]

        nodeH = row_normed_adj @ inputs

        """
        nodeH: (n, c)
        y: (n, c)
        """
        H = torch.stack([torch.mean(nodeH[torch.where(y_onehot[:, i])[0]], axis=0)
                         for i in range(y_onehot.shape[1])])
        assert H.shape[0] == y_onehot.shape[1]
        assert H.shape[1] == y_onehot.shape[1]

        H_nan = torch.isnan(H)

        if torch.any(H_nan):
            H = torch.where(H_nan, H.T, H)
            H_nan = torch.isnan(H)

        if torch.any(H_nan):
            H = torch.where(H_nan, torch.zeros_like(H), H)
            H_miss = (1 - torch.sum(H, axis=1, keepdims=True))
            H_miss /= torch.sum(H_nan, axis=1, keepdims=True)
            H = torch.where(H_nan, H_miss, H)
        H = self.makeDoubleStochasticH(H, max_iterations=3000)
        return H