from utils import *
from torch import optim
from copy import deepcopy
import argparse
from options import *
from torch.utils.data import DataLoader
import random
from sklearn.metrics import accuracy_score
from workflow import *
import IPython
from crowd_gan import *
from sampler import *
import copy
import itertools
from gcn.utils import *
from collections import Counter

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

dataset = 'labelme'
model_dir = './model/'

train_dataset = Dataset(mode='train', dataset=dataset, sparsity=0., k = 0)
trn_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

valid_dataset = Dataset(mode='valid', dataset=dataset)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

test_dataset = Dataset(mode='test', dataset=dataset)
tst_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


def pretrain_classifier(model, trace_scale=0):
    user_feature = np.eye(train_dataset.num_users)
    best_valid_acc = 0
    best_model = None
    lr = 1e-2
    train_acc_list = []
    test_acc_list = []
    for epoch in range(40):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_acc = train(train_loader=trn_loader, model=model, optimizer=optimizer, criterion=multi_loss, mode='crowdlayer', scale=trace_scale)
        valid_acc, valid_f1, _ = test(model=model, test_loader=val_loader)
        test_acc, test_f1, _ = test(model=model, test_loader=tst_loader)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = deepcopy(model)
        print('Epoch [%3d], Valid acc: %.5f, Valid f1: %.5f' % (epoch, valid_acc, valid_f1))
        print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))

    test_acc, test_f1, _ = test(model=best_model, test_loader=tst_loader)
    print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))
    torch.save(best_model.base_model.state_dict(), './model/generator_%s_pretrain' % dataset)
    return best_model


def pretrain_generator(generator):
    opt_G = torch.optim.Adam(generator.parameters(), lr= 1e-4)
    num_users = 59
    for _ in range(50):
        for idx, input, targets, targets_onehot, true_labels in trn_loader:
            input = input.cuda()
            true_labels = true_labels.cuda()
            targets = targets.cuda()
            
            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (input.shape[0] * num_users, 8)))).cuda()
            cls_out, gen_anno = generator(input, 'train')
            cls_out = cls_out.detach()
            anno_dist = generator.generate(x=input, latent_code=cls_out, noise=noise)
            anno_dist = anno_dist.reshape(cls_out.shape[0], num_users, -1)
            gen_anno = anno_dist.transpose(1, 2)
            g_loss = multi_loss(targets, gen_anno)
            
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            print(g_loss.item())
    print('Pre-train finished...')

def main(opt, gen_model, dis_model):
    train_acc_list = []
    test_acc_list = []

    user_feature = np.eye(train_dataset.num_users)
    generator = Generator(num_annotators=train_dataset.num_users, num_class=train_dataset.num_classes,
                       input_dims=train_dataset.input_dims, user_feature=user_feature, trace_reg=False).cuda()
    num_classes = train_dataset.num_classes 

    if gen_model:
        generator.load_state_dict(torch.load('./model/generator_%s_pretrain' % dataset))
    else:
        generator = pretrain_classifier(generator, trace_scale=0.)
        # pretrain_generator(generator)
        torch.save(generator.state_dict(), './model/generator_%s_pretrain' % dataset)

    discriminator = Discriminator()
    label_adj = np.load('./gcn/label_adj_labelme.npy')
    label_adj = normalize(label_adj + sp.eye(label_adj.shape[0]))
    label_adj = torch.from_numpy(np.array(label_adj)).cuda()

    num_items = len(train_dataset.y)
    discriminator.mlp_init(num_users=train_dataset.num_users, num_items=num_items, num_classes=train_dataset.num_classes, nb=2,
                           emb_dim=32, dropout=opt.dropout, item_embed=train_dataset.X, label_adj=label_adj)
    discriminator = discriminator.cuda()
    teacher_discriminator = copy.deepcopy(discriminator)
    teacher_generator = copy.deepcopy(generator)
    sampler = Sampler()

    if dis_model:
        discriminator.load_state_dict(torch.load('./model/discriminator_%s_pretrain' % dataset))
    else:
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0)
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
        optimizer_G_mat = torch.optim.Adam(generator.parameters(), lr=0)
        for _ in range(10):
            train_acc, anno_acc, true_acc, fake_acc, entropy, auc, clf_entropy = train_with_noise(train_loader=trn_loader, generator=generator,
                              discriminator=discriminator, opt_G=optimizer_G, opt_D=optimizer_D, opt=opt,
                                                            support=None, support_t=None, policy_grad=False, trace_scale=0.,
                                                            sampler=sampler, num_classes=num_classes, opt_G_M=optimizer_G_mat,
                                                            num_users=num_annotators)
        torch.save(discriminator.state_dict(), './model/discriminator_%s_pretrain' % dataset)

    optimizer_G = torch.optim.Adam(generator.base_model.parameters(), lr= opt.gen_lr / 5)
    optimizer_G_mat = torch.optim.Adam(generator.parameters(), lr=opt.gen_lr * 10)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.dis_lr, weight_decay=1e-2)
    
    best_valid_acc = 0
    best_model = None
    num_users = 59
    for epoch in range(opt.num_epochs):
        train_acc, anno_acc, true_acc, fake_acc, entropy, auc, clf_entropy = train_with_noise(train_loader=trn_loader, generator=generator,
                              discriminator=discriminator, opt_G=optimizer_G, opt_D=optimizer_D, opt=opt,
                                                            support=None, support_t=None, policy_grad=False, trace_scale=0.,
                                                            sampler=sampler, num_classes=num_classes, opt_G_M=optimizer_G_mat,
                                                            num_users=num_users)
        valid_acc, valid_f1, _ = test(model=generator, test_loader=val_loader)
        test_acc, test_f1, _ = test(model=generator, test_loader=tst_loader)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = deepcopy(generator)
        print('Epoch [%3d], Valid acc: %.5f, Valid f1: %.5f' % (epoch, valid_acc, valid_f1))
        print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))
    torch.save(best_model, model_dir + 'model_gan_%s' % dataset)
    test_acc, test_f1, _ = test(model=best_model, test_loader=tst_loader)
    print('Test acc: %.5f, Test f1: %.5f' % (test_acc, test_f1))
    return best_model, test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_opts(parser)
    opt = parser.parse_args()
    main(opt, gen_model=False, dis_model=True)