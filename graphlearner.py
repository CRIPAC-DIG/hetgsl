import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, epsilon=None, num_pers=16):
        super(GraphLearner, self).__init__()
        self.epsilon = epsilon

        self.weight_tensor = torch.Tensor(num_pers, input_size)
        self.weight_tensor = nn.Parameter(
            nn.init.xavier_uniform_(self.weight_tensor))

    def forward(self, input):
        """
        功能: 根据 metric 计算得到 $A^{(t)}$

        for IDGL:
            input: (n, d)
        """
        expand_weight_tensor = self.weight_tensor.unsqueeze(1)  # (m, 1, d)

        input_fc = input.unsqueeze(0) * expand_weight_tensor #(1, n, d)
        input_norm = F.normalize(input_fc, p=2, dim=-1) # 
        attention = torch.matmul(
            input_norm, input_norm.transpose(-1, -2)).mean(0)  # (n, n)
        # markoff_value = 0

        # attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)
        # return attention
        return attention

    def build_epsilon_neighbourhood(self, attention, markoff_value=0):
        mask = (attention > self.epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix
