from losses import *
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
import IPython
from copy import deepcopy
from sampler import *
from utils import *
from collections import Counter
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable
import torch.nn as nn

loss_fn = torch.nn.CrossEntropyLoss(reduce='mean').cuda()

def train(train_loader, model, optimizer, criterion=F.cross_entropy, mode='simple', annotators=None, pretrain=None,
          support = None, support_t = None, scale=0):
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    loss = 0

    correct_rec = 0
    total_rec = 0

    corr_anno = 0
    total_anno = 0

    for idx, input, targets, targets_onehot, true_labels in train_loader:
        if annotators != None:
            targets_onehot = targets_onehot[:, :, annotators]
        input = input.cuda()
        targets = targets.cuda().long()
        targets_onehot = targets_onehot.cuda()
        true_labels = true_labels.cuda().long()
        if mode == 'simple':
            loss = 0
            _, predicted = cls_out.max(1)
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)
        elif mode == 'true':
            output, _ = model(input)
            loss = loss_fn(output, true_labels)
            _, predicted = output.max(1)
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)
        elif mode == 'crowdlayer':
            cls_out, output = model(input)
            loss = criterion(targets, output)
            cls_prob = torch.clamp(cls_out, 1e-4, 1-1e-4)
            _, predicted = cls_out.max(1)
            correct += predicted.eq(true_labels).sum().item()
            total += true_labels.size(0)

            pi_0, gen_anno_idx = output.max(1)
            gen_anno_idx = gen_anno_idx[targets != -1]
            corr_anno += gen_anno_idx.eq(targets[targets != -1]).sum().item()
            total_anno += len(gen_anno_idx)

        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Training acc: ', correct / total)
    print('Training loss: ', total_loss.item())
    return correct / total


def re_index(raw_idx, dic):
    raw_idx = raw_idx.cpu().numpy()
    idx = [dic[i] for i in raw_idx]
    return torch.tensor(idx).cuda()


def train_with_noise(train_loader, generator, discriminator, opt_G, opt_D, opt=None, support=None, support_t=None, policy_grad=False, trace_scale=0,
             sampler=None, lambda_=0.01, anno_dist_real=None, teacher_generator=None, num_classes = 0, opt_G_C=None, train_dataset=None, 
             num_users=None, misgan=None, opt_mis=None, opt_G_M=None, opt_info=None):
    generator.train()
    discriminator.train()
    correct = 0
    total_ep = 0
    total_g_loss = 0
    total_d_loss = 0
    loss = 0

    correct = 0
    total = 0
    adversarial_loss = torch.nn.BCELoss(reduction='none')
    ce_loss_fn = torch.nn.CrossEntropyLoss(reduce='mean')
    continuous_loss = torch.nn.MSELoss(reduction='mean')
    bcelogit_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    corr_anno = 0
    total_anno = 0.1
    correct_true = 0
    total_true = 0.1
    correct_false = 0
    total_false = 0.1
    average_entropy = []
    entropy = []
    dis_auc = []

    np.random.seed(123)
    for idx, input, targets, targets_onehot, true_labels in train_loader:
        input = input.cuda()
        true_labels = true_labels.cuda()

        valid = targets[targets != -1].detach().fill_(1.).cuda()
        fake = targets[targets == -1].detach().fill_(0.).cuda()
        targets = targets.cuda()

        valid_mask = targets != -1
        fake_mask = targets == -1
        re_idx = {i: d for i, d in enumerate(idx)}

        cls_out, gen_anno = generator(input, mode='train')
        cls_out = cls_out.detach()

        noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (cls_out.shape[0] * num_users, num_classes)))).cuda()
        anno_dist = generator.generate(x=input, latent_code=cls_out, noise=noise).detach()
        anno_dist = anno_dist.reshape(cls_out.shape[0], num_users, -1)
        anno_dist = anno_dist.transpose(1, 2)
        anno_dist = torch.softmax(anno_dist, 1)

        gen_prob, anno_idx = sampler.sample_full(anno_dist.transpose(1, 2).cpu().numpy(),
                                                 targets.cpu().numpy())

        anno_dist_clip = torch.clamp(anno_dist, 1e-4, 1 - 1e-4)
        entropy_anno = -torch.sum(anno_dist_clip * torch.log(anno_dist_clip), 1)

        v_idx, u_idx = torch.where((targets != -2))
        v_idx = re_index(v_idx, re_idx)

        anno4dis = sampler.select_wrt_entropy(anno_mat=targets.cpu().numpy(),
                                                sampled_mat=anno_idx, entropy=entropy_anno.cpu().numpy(), ratio=1, reverse=True)

        anno4dis = torch.from_numpy(anno4dis).cuda()
        gen_prob = torch.from_numpy(gen_prob).cuda()
        anno_idx = torch.from_numpy(anno_idx).cuda()
        
        for _ in range(opt.dis_epoch):
            opt_D.zero_grad()
            v_idx_raw, u_idx = torch.where(targets != -1)
            v_idx = re_index(v_idx_raw, re_idx)
            type_real = targets[targets != -1]
            output_real, dist_real, latent_code = discriminator(u_idx, v_idx, type_real, policy_grad=True, act=nn.Softmax(1), 
                                                            v_idx_in_batch=v_idx_raw)

            score_real = torch.gather(output_real, index=type_real[:, None], dim=1).squeeze()

            info_loss = -torch.mean(torch.sum(cls_out[v_idx_raw] * torch.log_softmax(latent_code, 1), 1))
            real_loss = torch.mean(adversarial_loss(score_real, valid.float())) + opt.lambda_b * (torch.mean(output_real)) \
                             + opt.lambda_info * info_loss

            v_idx_raw, u_idx = torch.where((targets == -1) & (anno4dis != -1))
            v_idx = re_index(v_idx_raw, re_idx)
            type_fake = anno4dis[(targets == -1) & (anno4dis != -1)]

            fake_loss = 0
            selected = np.arange(len(type_fake))
            output_fake, dist_fake, latent_code = discriminator(u_idx[selected], v_idx[selected], type_fake[selected], policy_grad=True, 
                                                        act=nn.Softmax(1), v_idx_in_batch=v_idx_raw)

            score_fake = torch.gather(output_fake, index=type_fake[selected][:, None], dim=1).squeeze()
            info_loss = -torch.mean(torch.sum(cls_out[v_idx_raw] * torch.log_softmax(latent_code, 1), 1))
            fake_loss = torch.mean(adversarial_loss(score_fake, torch.zeros_like(type_fake).float())) + opt.lambda_b * (torch.mean(output_fake)) \
                                 + opt.lambda_info * info_loss
            fake_loss *= 0.9
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()
            
        v_idx, u_idx = torch.where((anno4dis != -2))
        true_v_idx = re_index(v_idx, re_idx)
        type_idx = anno_idx[(anno4dis != -2)]

        selected = np.arange(len(type_idx))
        reward, _, latent_code = discriminator(u_idx, true_v_idx, type_idx, policy_grad=True, act=nn.Softmax(1), 
                                            v_idx_in_batch=v_idx)
        
        info_reward = -torch.sum(cls_out[v_idx] * torch.log_softmax(latent_code, 1), 1)
        info_reward = info_reward.detach()
        info_reward_raw = info_reward.reshape(len(idx), -1)

        reward = reward.detach()
        
        reward_raw = torch.gather(reward, index=type_idx[:, None], dim=1).squeeze()
        reward = reward_raw.reshape(len(idx), -1)

        reward_clip = torch.clamp(reward, 1e-4, 1 - 1e-4)
        reward_entropy = -(reward_clip * torch.log(reward_clip) + (1 - reward_clip) * torch.log(1 - reward_clip))

        cls_prob = torch.clamp(cls_out, 1e-4, 1 - 1e-4)
        cls_ent = -torch.sum(cls_prob * torch.log(cls_prob), 1)
        average_entropy.append(cls_ent.mean().item())

        reward_ent_trh = 1.0
        ent_trh = 0.5

        ent_idx = cls_ent <= ent_trh
        ent_idx = ent_idx.repeat(num_users, 1).transpose(0, 1)

        ent_v_idx, ent_u_idx = torch.where((anno4dis != -2) & ent_idx & (reward_entropy <= reward_ent_trh))
        v_idx = re_index(ent_v_idx, re_idx)
        type_idx = anno_idx[(anno4dis != -2) & ent_idx & (reward_entropy <= reward_ent_trh)]

        pi_0 = gen_prob[ent_v_idx, ent_u_idx]
        reward = reward[ent_v_idx, ent_u_idx]
        info_reward = info_reward_raw[ent_v_idx, ent_u_idx]

        for _ in range(opt.gen_epoch):
            g_all = 0
            g_mat = 0

            anno_dist = generator.generate(x=input, latent_code=cls_out, noise=noise)
            anno_dist = anno_dist.reshape(cls_out.shape[0], num_users, -1)
            gen_anno_dist = anno_dist.transpose(1, 2)
            gen_anno_dist = torch.softmax(gen_anno_dist, 1)

            pi_w = gen_anno_dist[ent_v_idx, :, ent_u_idx]
            pi_w = torch.gather(pi_w, index=type_idx[:, None], dim=1).squeeze()

            log_reward = -torch.log((reward + 0.001) / (1 + 0.001))
            anno_ent = -torch.mean(torch.sum(torch.log(gen_anno_dist) * gen_anno_dist, 1))
            
            cf_ratio = pi_w / pi_0.float()
            cf_mat = cf_ratio
            reward_mat = log_reward

            r_ips = (reward_mat[cf_mat < opt.ips_threshold] - opt.lda) * cf_mat[cf_mat < opt.ips_threshold]
            g_mat += opt.lda + (torch.sum(r_ips) + torch.sum((reward_mat[cf_mat > opt.ips_threshold] - opt.lda) * opt.ips_threshold)) / len(cf_mat)
            
            info_ips = (info_reward[cf_mat < opt.ips_threshold] - opt.lda) * cf_mat[cf_mat < opt.ips_threshold]
            g_mat += opt.lambda_info * (opt.lda + (torch.sum(info_ips) + torch.sum((info_reward[cf_mat > opt.ips_threshold] 
                                                         - opt.lda) * opt.ips_threshold)) / len(cf_mat)) 
            opt_G_M.zero_grad()
            g_mat.backward()
            opt_G_M.step()

        target_fill = torch.ones_like(targets).cuda() * -1
        reward = reward_raw.reshape(targets.shape[0], targets.shape[1])
        target_fill[reward_entropy <= reward_ent_trh] = anno_idx[reward_entropy <= reward_ent_trh]

        target_fill = target_fill[cls_ent > ent_trh]
        v_idx, u_idx = torch.where(target_fill != -1)
        
        target_idx = target_fill[v_idx, u_idx]
        reward = reward[cls_ent > ent_trh]
        info_reward = info_reward_raw[cls_ent > ent_trh]

        reward = reward[v_idx, u_idx]
        info_reward = info_reward_raw[v_idx, u_idx]

        gen_prob = gen_prob[cls_ent > ent_trh]
        pi_0 = gen_prob[v_idx, u_idx]
        type_idx = target_fill[target_fill != -1]
        
        for _ in range(opt.gen_epoch):
            g_all = 0
            g_mat = 0
            if trace_scale:
                cls_out, gen_anno = generator(input, 'train')
                kernel = generator.base_model.kernel
                kernel = torch.softmax(kernel, 0)
                trace = 0
                for i in range(generator.base_model.num_annotators):
                    trace += torch.trace(kernel[:, :, i])
                g_all += trace_scale * trace
            else:
                cls_out, gen_anno = generator(input, 'train')

            anno_dist = generator.generate(x=input, latent_code=cls_out, noise=noise)
            anno_dist = anno_dist.reshape(cls_out.shape[0], num_users, -1)
            gen_anno = anno_dist.transpose(1, 2)
            gen_anno_dist = torch.softmax(gen_anno, 1)[cls_ent > ent_trh]

            cls_prob = torch.clamp(cls_out, 1e-4, 1 - 1e-4)[cls_ent > ent_trh]
            
            pi_w = gen_anno_dist[v_idx, :, u_idx]
            pi_w = torch.gather(pi_w, index=type_idx[:, None], dim=1).squeeze()

            log_reward = -torch.log((reward + 0.001) / (1 + 0.001))
            anno_ent = -torch.mean(torch.sum(torch.log(gen_anno_dist) * gen_anno_dist, 1))
            
            cf_ratio = pi_w / pi_0.float()
            cf_mat = cf_ratio
            reward_mat = log_reward

            r_ips = (reward_mat[cf_mat < opt.ips_threshold] - opt.lda) * cf_mat[cf_mat < opt.ips_threshold]
            g_all += opt.lda + (torch.sum(r_ips) + torch.sum((reward_mat[cf_mat > opt.ips_threshold] - opt.lda) * opt.ips_threshold)) / len(cf_mat)

            opt_G.zero_grad()
            g_all.backward()
            opt_G.step()

        _, predicted = cls_out.max(1)
        correct += predicted.eq(true_labels).sum().item()

        total += true_labels.size(0)
        total_d_loss += d_loss.item()
        total_ep += 1
    print('Training acc: ', correct / total)
    return correct / total, corr_anno / total_anno, correct_true / total_true, correct_false / total_false, np.mean(entropy), np.mean(dis_auc), None


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    target = []
    predict = []
    for _, inputs, targets in test_loader:
        inputs = inputs.cuda()
        target.extend(targets.data.numpy())
        targets = targets.cuda()

        total += targets.size(0)
        output, _ = model(inputs, mode='test')
        _, predicted = output.max(1)
        predict.extend(predicted.cpu().data.numpy())
        correct += predicted.eq(targets).sum().item()
    acc = correct / total
    f1 = f1_score(target, predict, average='macro')

    classes = list(set(target))
    classes.sort()
    acc_per_class = []
    predict = np.array(predict)
    target = np.array(target)
    for i in range(len(classes)):
        instance_class = target == i
        acc_i = np.mean(predict[instance_class] == classes[i])
        acc_per_class.append(acc_i)
    return acc, f1, acc_per_class


