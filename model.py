import pdb
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv

from util import normalize_sparse_tensor, row_sum_one_normalize
from layer import MLP, ChebNet, GCN, CompatibilityLayer
from graphlearner import GraphLearner


class CPGNN(nn.Module):
    def __init__(self, dataset, args):
        super(CPGNN, self).__init__()
        num_features = dataset['num_feature']
        out_dim = dataset['num_class']
        hidden = args.hidden
        if args.model == 'mlp':
            self.belief_estimator = MLP(num_features, hidden, out_dim, args.dropout)
        elif args.model == 'cheb':
            self.belief_estimator = ChebNet(num_features, hidden, out_dim, args.dropout)
        elif args.model == 'gcn':
            self.belief_estimator = GCN(num_features, hidden, out_dim, args.dropout)
        else:
            raise NotImplementedError('Belief estimator not specified, MLP or ChebNet ?')
        self.n_post_iter = args.n_post_iter
        self.H = nn.Parameter(torch.zeros(out_dim, out_dim))
        self.H_inited = False
        self.args = args
        self.dataset = dataset
        self.graph_learner1 = GraphLearner(num_features, hidden,
                                                epsilon=args.epsilon,
                                                num_pers=args.num_pers)

        self.graph_learner2 = GraphLearner(hidden,
                                                  out_dim,
                                                  epsilon=args.epsilon,
                                                 num_pers=args.num_pers)

    def learn_graph(self, learner:GraphLearner, features):

        raw_adj = learner(features)

        raw_adj = learner.build_epsilon_neighbourhood(raw_adj, markoff_value=0)
        adj = row_sum_one_normalize(raw_adj)
        adj = self.args.skip_conn * self.dataset['normed_adj'] + (1 - self.args.skip_conn) * adj

        return raw_adj, adj

    def forward_pretrain(self, normed_adj, features):
        """
        既不 graph learn, 也不 post-process
        """
        return self.belief_estimator(normed_adj, features)[0]


    def forward_one(self):
        """
        First iteration for graph learn
        """
        dataset = self.dataset
        x = dataset['features']
        raw_adj = dataset['raw_adj']
        y = dataset['labels']
        y_onehot = F.one_hot(y)
        cur_raw_adj, cur_normed_adj = self.learn_graph(self.graph_learner1, x)

        logits, node_vec = self.belief_estimator(cur_normed_adj, x)

        return logits, node_vec, cur_raw_adj, cur_normed_adj

    def forward_two(self, node_vec, first_adj):
        dataset = self.dataset
        x = dataset['features']
        raw_adj = dataset['raw_adj']
        y = dataset['labels'].cuda()
        y_onehot = F.one_hot(y)

        cur_raw_adj, cur_normed_adj = self.learn_graph(self.graph_learner2, node_vec)
        cur_normed_adj = self.args.update_ratio * cur_normed_adj + (1 - self.args.update_ratio) * first_adj
        logits, node_vec = self.belief_estimator(cur_normed_adj, x)

        return logits, node_vec, cur_raw_adj, cur_normed_adj        