import torch
import IPython
from scipy.special import logsumexp


def multi_loss(y_true, y_pred, loss_fn=torch.nn.CrossEntropyLoss(reduce='mean').cuda()):
    mask = y_true != -1
    y_pred = torch.transpose(y_pred, 1, 2)
    loss = loss_fn(y_pred[mask], y_true[mask])
    return loss


def ce_loss(target, output, loss_fn=torch.nn.KLDivLoss().cuda()):
    loss = 0
    for i in range(output.shape[2]):
        loss += loss_fn(target, output[:, :, 0])
    return loss


def mi_loss(target, output):
    loss = 0
    for i in range(output.shape[2]):
        pass
    pass