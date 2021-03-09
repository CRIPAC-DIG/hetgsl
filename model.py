import pdb
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F 

from util import normalize_sparse_tensor



class MLP(nn.Module):
    def __init__(self, num_features, hidden, out_dim, dropout):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(num_features, hidden)
        # pdb.set_trace()
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout
    def forward(self, input):
        x = self.lin1(input)
        x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return self.lin2(x)
        
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
            delta = torch.norm(H, p=1)
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
    def estimateH(self, adj, y, inputs=None, sample_mask=None):
        # RWNormAdj = (adj / tf.sparse.reduce_sum(adj, axis=1, keepdims=True))
        # raw_normed_adj = adj / adj.sum(dim=1, keepdims=True)
        # raw_normed_adj = adj / torch.sparse.sum(adj, dim=1)
        raw_normed_adj = normalize_sparse_tensor(adj)
        inputs = F.softmax(inputs, dim=1)
        inputs = inputs * (1 - sample_mask[:, None].float()) + y * sample_mask[:, None] # eq 10
        y = y * sample_mask[:, None]
        
        # nodeH = tf.sparse.sparse_dense_matmul(RWNormAdj, inputs) # eq 11
        nodeH = raw_normed_adj @ inputs

        # following to do
        # H = tf.concat([
        #     tf.reduce_mean(tf.gather(nodeH, tf.where(y[:, i]), axis=0), axis=0) for i in range(y.shape[1])
        # ], axis=0)
        """
        nodeH: (n, c)
        y: (n, c)
        """
        # H = torch.cat([torch.mean(torch.gather(nodeH, 0, torch.where(y[:, i])), axis=0) for i in range(y.shape[1])], axis=0)
        H = torch.stack([torch.mean(nodeH[torch.where(y[:, i])[0]], axis=0) for i in range(y.shape[1])])
        assert H.shape[0] == y.shape[1]
        assert H.shape[1] == y.shape[1]        

        # H_nan = tf.math.is_nan(H)
        H_nan = torch.isnan(H)

        # if tf.reduce_any(H_nan):
        #     H = tf.where(H_nan, tf.transpose(H), H)
        #     H_nan = tf.math.is_nan(H)
        if torch.any(H_nan):
            H = torch.where(H_nan, H.T, H)
            H_nan = torch.isnan(H)

        # if tf.reduce_any(H_nan):
        #     H = tf.where(H_nan, 0, H)
        #     H_miss = (1 - tf.reduce_sum(H, axis=1, keepdims=True))
        #     H_miss /= tf.reduce_sum(tf.cast(H_nan, H.dtype),
        #                             axis=1, keepdims=True)
        #     H = tf.where(H_nan, H_miss, H)
        # pdb.set_trace()
        if torch.any(H_nan):
            H = torch.where(H_nan, torch.zeros_like(H), H)
            H_miss = (1 - torch.sum(H, axis=1, keepdims=True))
            H_miss /= torch.sum(H_nan, axis=1, keepdims = True)
            H = torch.where(H_nan, H_miss, H)
        H = self.makeDoubleStochasticH(H, max_iterations=3000)
        return H

class LinBP(nn.Module):
    def __init__(self, iterations):
        super(LinBP, self).__init__()
        self.iterations = iterations
        # self.H_hat = nn.Parameter(int(y_train.shape[1]), int(y_train.shape[1]))
        self.H_hat = None
        

    def forward(self, inputs, adj, y_train, train_mask):

        if self.H_hat is None:
            H_init = CompatibilityLayer.estimateH(adj, y_train, inputs, train_mask)
            H_init = CompatibilityLayer.makeSymmetricH(H_init)
            H_init -= (1 / y_train.shape[1])
            self.H_hat = nn.Parameter(H_init)

        


        prior_belief = F.softmax(inputs, dim=1) # eq 4
        E_hat = prior_belief - (1 / y_train.shape[1]) # eq 5
        B_hat = E_hat



        for i in range(self.iterations):
            B_hat = E_hat + adj @ (B_hat @ self.H_hat) # eq 6
           
        post_belief = B_hat + (1 / y_train.shape[1]) # eq 7, accoring to open-sourced code, but different from the paper



        # self.add_loss(self.zero_reg_weight * tf.linalg.norm(
        #     tf.reduce_sum(self.non_linear_H(self.H_hat), axis=-1), ord=1)) # eq 12

        return post_belief


class CPGNN(nn.Module):
    def __init__(self, num_features, hidden, out_dim, args):
        super(CPGNN, self).__init__()
        self.belief_estimator = MLP(num_features, hidden, out_dim, args.dropout)
        self.linbp = LinBP(args.iterations)

    def forward(self, ipnuts, adj, y, train_mask):
        # pdb.set_trace()
        R = self.belief_estimator(ipnuts)
        # prior = F.softmax(R, dim=1)
        output = self.linbp(R, adj, y, train_mask)

        return output
