import pdb
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, ChebConv

from util import normalize_sparse_tensor


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


class ChebNet(nn.Module):
    def __init__(self, num_features, hidden, out_dim, dropout):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(num_features, hidden, K=2)
        self.conv2 = ChebConv(hidden, out_dim, K=2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index, edge_weight)
        return x

        
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
    def estimateH(self, adj, y, init_inputs=None, sample_mask=None):
        raw_normed_adj = normalize_sparse_tensor(adj) # make row sum to 1
        inputs = F.softmax(init_inputs, dim=1)
        inputs = inputs * (1 - sample_mask[:, None].float()) + y * sample_mask[:, None] # eq 10
        y = y * sample_mask[:, None]
        
        nodeH = raw_normed_adj @ inputs

        """
        nodeH: (n, c)
        y: (n, c)
        """
        H = torch.stack([torch.mean(nodeH[torch.where(y[:, i])[0]], axis=0) for i in range(y.shape[1])])
        assert H.shape[0] == y.shape[1]
        assert H.shape[1] == y.shape[1]        

        H_nan = torch.isnan(H)

        if torch.any(H_nan):
            H = torch.where(H_nan, H.T, H)
            H_nan = torch.isnan(H)

        if torch.any(H_nan):
            H = torch.where(H_nan, torch.zeros_like(H), H)
            H_miss = (1 - torch.sum(H, axis=1, keepdims=True))
            H_miss /= torch.sum(H_nan, axis=1, keepdims = True)
            H = torch.where(H_nan, H_miss, H)
        H = self.makeDoubleStochasticH(H, max_iterations=3000)
        return H

class LinBP(nn.Module):
    def __init__(self, iterations, out_dim):
        super(LinBP, self).__init__()
        self.iterations = iterations
        self.H_hat = nn.Parameter(torch.zeros(out_dim, out_dim))
        self.inited = False
        
    def forward(self, inputs, adj, y_train, train_mask):
        if not self.inited:
            with torch.no_grad():
                H_init = CompatibilityLayer.estimateH(adj, y_train, inputs, train_mask)
                H_init = CompatibilityLayer.makeSymmetricH(H_init)
                H_init -= (1 / y_train.shape[1])
                self.H_hat.data = H_init
            self.inited = True

        
        prior_belief = F.softmax(inputs, dim=1) # eq 4
        E_hat = prior_belief - (1 / y_train.shape[1]) # eq 5, E_hat is B^0
        B_hat = E_hat



        for i in range(self.iterations):
            B_hat = E_hat + adj @ (B_hat @ self.H_hat) # eq 6
           
        post_belief = B_hat + (1 / y_train.shape[1]) # eq 7, accoring to open-sourced code, but different from the paper

        reg_h_loss = torch.norm(self.H_hat.sum(dim=1), p=1)

        return post_belief, reg_h_loss


class CPGNN(nn.Module):
    def __init__(self, num_features, hidden, out_dim, args):
        super(CPGNN, self).__init__()
        if args.mlp:
            self.belief_estimator = MLP(num_features, hidden, out_dim, args.dropout)
        elif args.cheb:
            self.belief_estimator = ChebNet(num_features, hidden, out_dim, args.dropout)
        else:
            raise NotImplementedError('Belief estimator not specified, MLP or ChebNet ?')
        self.linbp = LinBP(args.iterations, out_dim)

    def forward(self, data, inputs, adj, y, train_mask):
        # pdb.set_trace()
        edge_index = data.edge_index.cuda()
        if data.edge_attr is not None:
            edge_weight = data.attr.cuda()
        else:
            edge_weight = None
        R = self.belief_estimator(inputs, edge_index, edge_weight)
        # prior = F.softmax(R, dim=1)
        post_belief, reg_h_loss = self.linbp(R, adj, y, train_mask)

        return post_belief, R, reg_h_loss

    def forward_one(self, data, inputs, adj, y, train_mask):
        edge_index = data.edge_index.cuda()
        if data.edge_attr is not None:
            edge_weight = data.attr.cuda()
        else:
            edge_weight = None
        return self.belief_estimator(inputs, edge_index, edge_weight)