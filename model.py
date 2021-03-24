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
        # self.linbp = LinBP(args.iterations, out_dim)
        self.n_post_iter = args.n_post_iter
        self.H = nn.Parameter(torch.zeros(out_dim, out_dim))
        self.H_inited = False
        self.args = args
        self.dataset = dataset
    
        if args.graph_learn:
            self.graph_learner1 = GraphLearner(num_features, hidden,
                                                 epsilon=args.epsilon,
                                                 num_pers=args.num_pers)

            self.graph_learner2 = GraphLearner(hidden,
                                                  out_dim,
                                                  epsilon=args.epsilon,
                                                 num_pers=args.num_pers)

    def learn_graph(self, learner:GraphLearner, features, train_mask, mulH=False, logits=None, y_onehot=None):

        raw_adj = learner(features)

        if mulH:
            assert logits is not None
            assert y_onehot is not None
            pred = F.softmax(logits, dim=1)
            y_train = pred * (1 - train_mask[:, None].float()) + y_onehot * train_mask[:, None]
            raw_adj = raw_adj * (y_train @ self.H @ y_train.T)
        
        raw_adj = learner.build_epsilon_neighbourhood(raw_adj, markoff_value=0)
        adj = row_sum_one_normalize(raw_adj)
        adj = self.args.skip_conn * self.dataset['normed_adj'] + (1 - self.args.skip_conn) * adj

        return raw_adj, adj

    def forward_pretrain(self, normed_adj, features):
        """
        既不 graph learn, 也不 post-process
        """
        return self.belief_estimator(normed_adj, features)[0]

    def forward(self, raw_adj, normed_adj, features, y_onehot, train_mask):
        """
        post_process
        """
        logits = self.belief_estimator(normed_adj, features)[0]
        # post_belief, reg_h_loss = self.linbp(logits, raw_adj, y_onehot, train_mask)
        if not self.H_inited:
            self.init_H(raw_adj, y_onehot, logits, train_mask)
        post_belief = self.post_process(raw_adj, logits, y_onehot)
        return post_belief

    def forward_one(self, logits, train_mask):
        """
        First iteration for graph learn
        """
        dataset = self.dataset
        x = dataset['features']
        # normed_adj = dataset['normed_adj'].cuda()
        raw_adj = dataset['raw_adj']
        y = dataset['labels']
        y_onehot = F.one_hot(y)
        if self.args.mulH and not self.H_inited:
            self.init_H(raw_adj, y_onehot, logits, train_mask)
        
        cur_raw_adj, cur_normed_adj = self.learn_graph(self.graph_learner1, x, train_mask, mulH=self.args.mulH, logits=logits, y_onehot=y_onehot)

        logits, node_vec = self.belief_estimator(cur_normed_adj, x)

        if self.args.post:
            logits = self.post_process(raw_adj, logits, y_onehot)

        return logits, node_vec, cur_raw_adj, cur_normed_adj

    def forward_two(self, node_vec, logits, train_mask, first_adj):
        dataset = self.dataset
        x = dataset['features']
        # normed_adj = dataset['normed_adj'].cuda()
        raw_adj = dataset['raw_adj']
        y = dataset['labels'].cuda()
        y_onehot = F.one_hot(y)

        if self.args.mulH:
            assert self.H_inited

        cur_raw_adj, cur_normed_adj = self.learn_graph(self.graph_learner2, node_vec, train_mask, mulH=self.args.mulH, logits=logits, y_onehot=y_onehot)
        cur_normed_adj = self.args.update_ratio * cur_normed_adj + (1 - self.args.update_ratio) * first_adj
        logits, node_vec = self.belief_estimator(cur_normed_adj, x)

        if self.args.post:
            logits = self.post_process(raw_adj, logits, y_onehot)

        return logits, node_vec, cur_raw_adj, cur_normed_adj
        

    def init_H(self, raw_adj, y_onehot, logits, train_mask):
        assert not self.H_inited
        print('Initing H...')
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