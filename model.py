import pdb
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv

from util import normalize_sparse_tensor, row_sum_one_normalize
from layer import MLP, ChebNet, GCN, LinBP
from graphlearner import GraphLearner


class CPGNN(nn.Module):
    def __init__(self, num_features, hidden, out_dim, args):
        super(CPGNN, self).__init__()
        if args.model == 'mlp':
            self.belief_estimator = MLP(num_features, hidden, out_dim, args.dropout)
        elif args.model == 'cheb':
            self.belief_estimator = ChebNet(num_features, hidden, out_dim, args.dropout)
        elif args.model == 'gcn':
            self.belief_estimator = GCN(num_features, hidden, out_dim, args.dropout)
        else:
            raise NotImplementedError('Belief estimator not specified, MLP or ChebNet ?')
        self.linbp = LinBP(args.iterations, out_dim)
    
        if args.graph_learn:
            self.graph_learner1 = GraphLearner(num_features, hidden,
                                                 epsilon=args.epsilon,
                                                 num_pers=args.num_pers)

            self.graph_learner2 = GraphLearner(hidden,
                                                  out_dim,
                                                  epsilon=args.epsilon,
                                                 num_pers=args.num_pers)

    def forward(self, raw_adj, normed_adj, features, y_onehot, train_mask):
        """
        use linbp
        """
        R = self.belief_estimator(normed_adj, features)
        post_belief, reg_h_loss = self.linbp(R, raw_adj, y_onehot, train_mask)

        return post_belief, R, reg_h_loss

    def forward_pretrain(self, normed_adj, features):
        """
        pretrain
        """
        return self.belief_estimator(normed_adj, features)
