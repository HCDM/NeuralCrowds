import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import IPython
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True, num_annotators=59):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(num_annotators, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = input.reshape(input.shape[0], -1)

        if len(adj.shape) == 2:
            output = torch.mm(adj.float(), input)
        elif len(adj.shape) == 3:
            output = torch.einsum('ijk,kl->ijl', adj.float(), input)
            output += input.repeat(adj.shape[0], 1, 1)

        if input.shape[1] == output.shape[1]:
            output += input

        if self.bias is not None:
            return output + self.bias[:, None, :]
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'