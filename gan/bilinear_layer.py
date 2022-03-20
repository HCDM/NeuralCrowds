import torch.nn as nn
import torch.nn.functional as F
import torch

class Bilinear_Layer(nn.Module):
    def __init__(self, dropout=0.7, act=torch.nn.Softmax):
        super(Bilinear_Layer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, u_hidden, v_hidden, type_idx, weights, policy_grad=False):
        u_hidden = self.dropout(u_hidden)
        v_hidden = self.dropout(v_hidden)

        basis_outputs = []
        for weight in weights:
            u_w = torch.matmul(u_hidden, weight)
            x = torch.sum(torch.mul(u_w, v_hidden), 1)
            basis_outputs.append(x)
        outputs = torch.stack(basis_outputs, 1)
        if policy_grad:
            pass
        else:
            outputs = torch.gather(outputs, dim=1, index=type_idx[:, None])
        outputs = self.act(outputs)
        return outputs.squeeze()

