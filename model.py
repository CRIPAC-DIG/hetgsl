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
    
        if args.graph_learn:
            self.graph_learner1 = GraphLearner(num_features, hidden,
                                                 epsilon=args.epsilon,
                                                 num_pers=args.num_pers)

            self.graph_learner2 = GraphLearner(hidden,
                                                  out_dim,
                                                  epsilon=args.epsilon,
                                                 num_pers=args.num_pers)

    def learn_graph(self, learner:GraphLearner, features, train_mask, first_adj=None, mulH=False, logits=None, y_onehot=None):

        raw_adj = learner(features)

        if mulH:
            assert logits is not None
            assert y_onehot is not None
            pred = F.softmax(logits, dim=1)
            y_train = pred * (1 - train_mask[:, None].float()) + y_onehot * train_mask[:, None]
            H_ratio = self.args.H_ratio
            H_coef = y_train @ self.H @ y_train.T
            E_coef = torch.ones_like(H_coef, device=H_coef.device)
            raw_adj = raw_adj * (H_ratio * H_coef + (1-H_ratio) * E_coef)
        
        raw_adj = learner.build_epsilon_neighbourhood(raw_adj, markoff_value=0)
        adj = row_sum_one_normalize(raw_adj)

        if first_adj is not None:
            alpha, beta = self.args.alpha, self.args.beta
            adj = alpha * self.dataset['normed_adj'] + beta * first_adj + (1-alpha - beta) * adj 

        return adj

    def forward_pretrain(self, normed_adj, features):
        """
        既不 graph learn, 也不 post-process
        """
        return self.belief_estimator(normed_adj, features)[0]

    def forward_cpgnn(self, train_mask):
        dataset = self.dataset
        x = dataset['features']
        normed_adj = dataset['normed_adj'].cuda()
        raw_adj = dataset['raw_adj']
        y = dataset['labels']
        y_onehot = F.one_hot(y)

        logits, node_vec = self.belief_estimator(normed_adj, x)
        
        if not self.H_inited:
            self.init_H(raw_adj, y_onehot, logits, train_mask)

        logits = self.post_process(raw_adj, logits, y_onehot)

        return logits

    def forward_one(self, train_mask):
        """
        First iteration for graph learn
        """
        dataset = self.dataset
        x = dataset['features']
        normed_adj = dataset['normed_adj'].cuda()
        raw_adj = dataset['raw_adj']
        y = dataset['labels']
        y_onehot = F.one_hot(y)

        logits, node_vec = self.belief_estimator(normed_adj, x)

        if self.args.mulH and not self.H_inited:
            self.init_H(raw_adj, y_onehot, logits, train_mask)        
        
        if self.args.post:
            logits = self.post_process(raw_adj, logits, y_onehot)
        
        first_adj = self.learn_graph(self.graph_learner1, x, train_mask, first_adj=None, mulH=self.args.mulH, logits=logits, y_onehot=y_onehot)

        return logits, node_vec, first_adj

    def forward_two(self, node_vec, logits, train_mask, first_adj):
        dataset = self.dataset
        x = dataset['features']
        raw_adj = dataset['raw_adj']
        y = dataset['labels'].cuda()
        y_onehot = F.one_hot(y)

        if self.args.mulH:
            assert self.H_inited

        cur_normed_adj = self.learn_graph(self.graph_learner2, node_vec, train_mask, first_adj=first_adj, mulH=self.args.mulH, logits=logits, y_onehot=y_onehot)
        logits, node_vec = self.belief_estimator(cur_normed_adj, x)

        if self.args.post:
            logits = self.post_process(raw_adj, logits, y_onehot)

        return logits, node_vec
        

    def init_H(self, raw_adj, y_onehot, logits, train_mask):
        # pdb.set_trace() 

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
