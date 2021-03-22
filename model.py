import pdb
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv

from util import normalize_sparse_tensor, row_sum_one_normalize
from layer import MLP, ChebNet, GCN, LinBP, CompatibilityLayer
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
        # self.linbp = LinBP(args.iterations, out_dim)
        self.n_post_iter = args.n_post_iter
        self.H = nn.Parameter(torch.zeros(out_dim, out_dim))
        self.H_inited = False
    
        if args.graph_learn:
            self.graph_learner1 = GraphLearner(num_features, hidden,
                                                 epsilon=args.epsilon,
                                                 num_pers=args.num_pers)

            self.graph_learner2 = GraphLearner(hidden,
                                                  out_dim,
                                                  epsilon=args.epsilon,
                                                 num_pers=args.num_pers)

    def learn_graph(self, learner:GraphLearner, features, train_mask, graph_skip_conn, init_normed_adj, mulH=False, pred=None, y_onehot=None):

        raw_adj = learner(features)

        if mulH:
            assert pred is not None
            assert y_onehot is not None
            y_train = pred * (1 - train_mask[:, None].float()) + y_onehot * train_mask[:, None]
            raw_adj = raw_adj * (y_train @ self.H @ y_train.T)
        
        raw_adj = learner.build_epsilon_neighbourhood(raw_adj, markoff_value=0)
        adj = row_sum_one_normalize(raw_adj)
        adj = graph_skip_conn * init_normed_adj + (1 - graph_skip_conn) * adj

        return raw_adj, adj

    def forward_pretrain(self, normed_adj, features):
        """
        Pretrain
        """
        return self.belief_estimator(normed_adj, features)

    def forward(self, raw_adj, normed_adj, features, y_onehot, train_mask):
        """
        Use LinBP without graph learn
        """
        logits = self.belief_estimator(normed_adj, features)
        # post_belief, reg_h_loss = self.linbp(logits, raw_adj, y_onehot, train_mask)
        if not self.H_inited:
            self.init_H(raw_adj, y_onehot, logits, train_mask)
        post_belief = self.post_process(raw_adj, logits, y_onehot)
        return post_belief

    def forward_one(self, node_features, ):
        """
        First iteration for graph learn
        """
        pass

    def forward_two(self,):
        """
        Remaing iterations for graph learn
        """
        pass



    def init_H(self, raw_adj, y_onehot, logits, train_mask):
        assert not self.H_inited
        with torch.no_grad():
            H_init = CompatibilityLayer.estimateH(
                raw_adj, y_onehot, logits, train_mask)
            H_init = CompatibilityLayer.makeSymmetricH(H_init)
            H_init -= (1 / y_onehot.shape[1])
            self.H.data = H_init
        self.H_inited = True

    def post_process(self, raw_adj, logits, y_onehot):
        assert self.H_inited
        prior_belief = F.softmax(logits, dim=1)  # eq 4
        E_hat = prior_belief - (1 / y_onehot.shape[1])  # eq 5, E_hat is B^0
        B_hat = E_hat

        for _ in range(self.n_post_iter):
            B_hat = E_hat + raw_adj @ (B_hat @ self.H)  # eq 6

        # eq 7, accoring to open-sourced code, but different from the paper
        post_belief = B_hat + (1 / y_onehot.shape[1])

        return post_belief