import argparse
import os
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from gcn.models import GCN
from bilinear_layer import *
from torch.autograd import Variable
import IPython



def init_identities(shape, dtype=None):
    out = np.zeros(shape)
    for r in range(shape[2]):
        for i in range(shape[0]):
            out[i, i, r] = 2.0
    return torch.Tensor(out).cuda()

class Base_Model(nn.Module):
    def __init__(self, num_annotators, input_dims, num_class, dropout=0.5,
                 backbone_model=None, emb_dim=128):
        super(Base_Model, self).__init__()
        self.num_annotators = num_annotators

        self.bn1 = torch.nn.BatchNorm1d(input_dims, affine=False)
        self.bn2 = torch.nn.BatchNorm1d(emb_dim, affine=False)

        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.kernel = nn.Parameter(init_identities((num_class, num_class, num_annotators)),
                                   requires_grad=True)
        self.backbone_model = None
        if backbone_model == 'resnet34':
            self.backbone_model = ResNet34()
        if backbone_model == 'vgg16':
            self.backbone_model = VGG('VGG16').cuda()
        if backbone_model == 'cnn':
            self.backbone_model = CNN(input_channel=1, n_outputs=10).cuda()
        
        self.classifier = nn.Sequential(
                                        # nn.BatchNorm1d(input_dims, affine=False),
                                        nn.Linear(input_dims, emb_dim), 
                                        nn.ReLU(), nn.Dropout(dropout), 
                                        # nn.BatchNorm1d(emb_dim, affine=False),
                                        nn.Linear(emb_dim, num_class), 
                                        nn.Softmax(dim=1))

    def forward(self, x, mode=None):
        cls_out = 0
        if self.backbone_model:
            cls_out = self.backbone_model(x)
        else:
            x = x.view(x.size(0), -1)
            cls_out = self.classifier(x)
        crowd_out = torch.einsum('ij,jkl->ikl', (cls_out, self.kernel))
        return cls_out, crowd_out


class Generator(nn.Module):
    def __init__(self, num_annotators, input_dims, num_class, dropout=0.5,
                 backbone_model=None, emb_dim=128, user_feature=None, trace_reg=False,
                 user_emb_dim=32):
        super(Generator, self).__init__()
        self.num_annotators = num_annotators
        self.gen_emb_dim = 20 

        self.user_emb = torch.eye(num_annotators).cuda()
        self.lin_user = nn.Linear(num_annotators, self.gen_emb_dim)

        self.num_class = num_class
        self.trace_reg = trace_reg

        self.base_model = Base_Model(num_annotators, input_dims, num_class,
                                 dropout=0.5, backbone_model=backbone_model, emb_dim=128)
        
        self.gen_flow = nn.Sequential(nn.Linear(num_class * 2 + self.gen_emb_dim * 2, 128),
                                        nn.ReLU(), nn.Linear(128, num_class))

        if backbone_model:
            self.extracter = VGG('VGG16').cuda().features
            self.item_emb_layer = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), 
                                                nn.Linear(512, self.gen_emb_dim))
        else:
            self.item_emb_layer = nn.Sequential(
                                            nn.Linear(input_dims, 64), 
                                            nn.ReLU(), nn.Dropout(dropout), 
                                            nn.BatchNorm1d(64, affine=False),
                                            nn.Linear(64, self.gen_emb_dim))      
        self.gen_bn = torch.nn.BatchNorm1d(self.gen_emb_dim * 2, affine=False)
        self.backbone_model = backbone_model
        

    def forward(self, x, mode='train'):
        if self.trace_reg and mode == 'train':
            cls_out, crowd_out, trace = self.base_model(x, mode=mode)
            return cls_out, crowd_out, trace
        else:
            cls_out, crowd_out = self.base_model(x, mode=mode)
            return cls_out, crowd_out

    def load_pretrain(self, model_path):
        self.base_model.load_state_dict(torch.load(model_path))
    
    def generate(self, x, noise, latent_code):
        if self.backbone_model:
            x = self.extracter(x)
        item_emb = self.item_emb_layer(x.squeeze())
        item_emb = torch.repeat_interleave(item_emb, self.num_annotators, dim=0)
        
        user_emb = self.lin_user(self.user_emb)
        user_emb = torch.cat(latent_code.shape[0] * [user_emb])

        latent_code = torch.repeat_interleave(latent_code, self.num_annotators, dim=0)
        new_code = torch.cat((user_emb, item_emb), -1)
        new_code = self.gen_bn(new_code)
        new_code = torch.cat((new_code, noise, latent_code), -1)
        return self.gen_flow(new_code)


class MLP_Discriminator(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, num_classes, nb, dropout, item_embed=None, label_adj=None, backbone_model=None):
        super(MLP_Discriminator, self).__init__()
        hidden_dim = 60
        self.item_emb = item_embed.cuda()
        self.num_users = num_users
        self.user_emb = torch.eye(num_users).cuda()

        self.class_emb = torch.eye(num_classes).cuda()
        self.bilinear = Bilinear_Layer(dropout=dropout, act=lambda x: x)

        self.W = nn.Parameter(torch.rand(num_classes, hidden_dim, hidden_dim), requires_grad=True)

        self.cls_idx = torch.eye(num_classes).cuda()
        self.label_gcn = GCN(nfeat=hidden_dim, nhid=hidden_dim, dropout=dropout, att_mode='gcn', num_annotators=num_users) 

        self.dropout = torch.nn.Dropout(dropout)

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        if backbone_model:
            self.v_init = VGG('VGG16').cuda().features
            self.lin_v = nn.Linear(512, hidden_dim)
            self.item_base = nn.Linear(512, num_classes)
        else:
            self.v_init = nn.Linear(item_embed.shape[1], 128)
            self.lin_v = nn.Linear(128, hidden_dim)
            self.item_base = nn.Linear(128, num_classes)

        self.lin_u = nn.Linear(num_users, hidden_dim)
        self.bn_v = nn.BatchNorm1d(hidden_dim)
        self.bn_u = nn.BatchNorm1d(hidden_dim)
        self.lin_c = nn.Linear(hidden_dim * hidden_dim, hidden_dim)

        self.adj = label_adj
        self.dp_rate = dropout

        self.aux_output = nn.Sequential(
                                    nn.Linear(hidden_dim * 3, 128),
                                    nn.ReLU(), nn.Linear(128, num_classes))

        self.aux_bn = nn.BatchNorm1d(hidden_dim * 3, affine=False)

    def forward(self, u_idx, v_idx, type, policy_grad=False, v_idx_in_batch=None):
        v_idx_uni = torch.unique(v_idx)
        v_emb = self.v_init(self.item_emb[v_idx_uni]).squeeze()

        u_hidden = self.lin_u(self.user_emb)
        v_hidden = self.lin_v(F.relu(v_emb))

        W = self.label_gcn(self.W, self.adj, u_hidden=u_hidden, v_hidden=v_hidden)
        W = W.reshape(self.num_classes, self.hidden_dim, self.hidden_dim)

        reidx = v_idx_uni.cpu().numpy()
        v_idx = v_idx.cpu().numpy()
        reidx = {d:i for i, d in enumerate(reidx)}
        v_idx = [reidx[i] for i in v_idx]

        u_hidden = u_hidden[u_idx]
        v_hidden = v_hidden[v_idx]
        validity = self.bilinear(u_hidden, v_hidden, type, weights = W, policy_grad=policy_grad)

        return validity
    
    def auxiliary_network(self, u_idx, v_idx, anno_idx):
        v_idx_uni = torch.unique(v_idx)
        v_emb = self.v_init(self.item_emb[v_idx_uni]).squeeze()

        u_hidden = self.lin_u(self.user_emb)
        v_hidden = self.lin_v(F.relu(v_emb))

        reidx = v_idx_uni.cpu().numpy()
        v_idx = v_idx.cpu().numpy()
        reidx = {d:i for i, d in enumerate(reidx)}
        v_idx = [reidx[i] for i in v_idx]

        u_hidden = u_hidden[u_idx]
        v_hidden = v_hidden[v_idx]

        W = self.label_gcn(self.W, self.adj, u_hidden=u_hidden, v_hidden=v_hidden)
        cls_hidden = self.lin_c(F.relu(W))
        cls_hidden = cls_hidden[anno_idx]

        hidden = torch.cat([u_hidden, v_hidden, cls_hidden], dim=1)
        hidden = self.aux_bn(hidden)
        aux_output = self.aux_output(hidden)
        return aux_output

class Discriminator(nn.Module):
    def __init__(self, opt=None):
        super(Discriminator, self).__init__()

    def mlp_init(self, num_users, num_items, emb_dim, num_classes, nb, dropout, item_embed, label_adj=None, backbone_model=None):
        self.discriminator = MLP_Discriminator(num_users, num_items, emb_dim, num_classes, nb, dropout, item_embed, label_adj=label_adj, backbone_model=backbone_model)

    def forward(self, u_idx, v_idx, anno_idx, support=None, support_t=None, policy_grad=True, act=None, v_idx_in_batch=None):
        validity_score = self.discriminator(u_idx, v_idx, anno_idx, policy_grad=policy_grad, v_idx_in_batch=v_idx_in_batch)
        validity = torch.sigmoid(validity_score)
        class_dist = act(validity_score)

        latent_code = self.discriminator.auxiliary_network(u_idx,v_idx, anno_idx)
        return validity, validity_score, latent_code
    
    def auxiliary_network(self, u_idx, v_idx, anno_dist):
        latent_code = self.discriminator.auxiliary_network(u_idx, v_idx, anno_dist)
        return latent_code