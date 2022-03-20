import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution
import IPython
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha=0.2, nheads=6, att_mode='gat', num_annotators = 59):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.att_mode = att_mode
        self.gc1 = GraphConvolution(nfeat * nfeat, nhid * nhid, bias=False)

    def forward(self, x, adj, u_hidden=None, v_hidden=None):
        x = x.reshape(x.shape[0], -1)
        x = self.gc1(x, adj)
        return x