# @Time    : 2021/3/12 10:47
# @FileName: covari.py
# @Software: PyCharm
from torch.autograd import Variable
import argparse
import os, sys
import os.path as osp
import torchvision
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
import datetime, time
import utils
from math import exp


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()
    txt_src = open(args.s_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                new_src.append(line)
        txt_src = new_src.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dsets["source"] = ImageList_idx(txt_src, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def trade_off_scheduler(iter_num, max_iter):
    p = iter_num / max_iter
    return 1-(2 / (1 + exp(- 10 * p)) - 1), np.exp(0.3 * p) - 1


def train_target(args):
    dset_loaders = data_load(args)
    avg_loss = utils.AvgLoss()
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
        netF1 = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bootleneck(feature_dim=netF.in_features, args=args, bottleneck_dim=args.bottleneck).cuda()
    netB1 = network.feat_bootleneck(feature_dim=netF.in_features, args=args, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(class_num=args.class_num, args=args, bottleneck_dim=args.bottleneck).cuda()
    netC1 = network.feat_classifier(class_num=args.class_num, args=args, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    netF1.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    netB1.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC1.load_state_dict(torch.load(modelpath))
    netC.eval()
    netB1.eval()
    netF1.eval()

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
    for k, v in netC1.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * args.lr_decay3}]

    for k, v in netC.named_parameters():
        v.requires_grad = False
    for k, v in netF1.named_parameters():
        v.requires_grad = False
    for k, v in netB1.named_parameters():
        v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.max_epoch
    iter_num = 0
    epoch_num = 0
    netF.eval()
    netB.eval()
    threshold = utils.obtain_label(dset_loaders['test'], netF, netB, netC, args, threshold=True)
    netF.train()
    netB.train()
    loss_list = []
    acc_list_ = []
    start_time = time.time()
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue
        inputs_test = inputs_test.cuda()
        if iter_num % interval_iter == 0:
            epoch_num += 1

        if iter_num % interval_iter == 0:
            netF.eval()
            netB.eval()
            mem_label = utils.obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()
        iter_num += 1
        optimizer = lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        features_src = netF1(inputs_test)
        features_tgt = netF(inputs_test)
        features_higher_src = netB1(features_src).detach()
        features_higher_tgt = netB(features_tgt)
        outputs_tgt = netC1(features_higher_tgt, reverse=True, alpha=1)
        outputs_src = netC1(features_higher_src)
        outputs_f_tgt = netC(features_higher_tgt)
        outputs_f_src = netC(features_higher_src)
        pred_f_tgt = outputs_f_tgt.argmax(dim=1)
        pred_f_src = outputs_f_src.argmax(dim=1)
        loss_tgt = torch.log(1 + 1e-6 - F.softmax(outputs_tgt, dim=1))
        loss_adv_tgt = F.nll_loss(loss_tgt, pred_f_tgt)
        loss_adv_src = loss.topk_loss(inputs_f=outputs_f_src, inputs=outputs_src, pred=pred_f_src, thres=threshold)

        adv_loss = loss_adv_tgt + loss_adv_src
        if args.dset == 'VISDA-C':
            if epoch_num == 1:
                loss_ = loss.TarDisClusterLoss(output=outputs_f_tgt)
            else:
                pred = mem_label[tar_idx]
                loss_ = args.adv * adv_loss + args.adv * nn.CrossEntropyLoss()(outputs_f_tgt, pred) + loss_clu
        else:
            pred = mem_label[tar_idx]
            loss_clu = loss.TarDisClusterLoss(output=outputs_f_tgt)
            loss_ = args.alpha * adv_loss + args.beta * nn.CrossEntropyLoss()(outputs_f_tgt,
                                                                              pred) + args.gamma * loss_clu

        loss_list.append(nn.CrossEntropyLoss()(outputs_f_tgt, _.cuda()).cpu().detach())
        avg_loss.add_loss(nn.CrossEntropyLoss()(outputs_f_tgt, _.cuda()).cpu().detach())
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            end_time = time.time()
            print('Time need: {:.1f}'.format(end_time - start_time))
            break

            netF.eval()
            netB.eval()
            loss__ = avg_loss.get_avg_loss()

            if args.dset == 'VISDA-C':
                acc_s_te, acc_list = utils.cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Loss: {:.2f}, Iter:[{}/{}]; Accuracy = {:.2f}%'.format(args.name, loss__, epoch_num, args.max_epoch,
                                                                            acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = utils.cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Loss: {:.2f}, Iter:[{}/{}]; Accuracy = {:.2f}%'.format(args.name, loss__, epoch_num, args.max_epoch, acc_s_te)
            acc_list_.append(acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
    netF.eval()
    netB.eval()
    netC.eval()

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SFADA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=2, help="source")
    parser.add_argument('--t', type=int, default=0, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=3e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2019)
    parser.add_argument('--T', type=float, default=3.0)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--beta', type=float, default=.3)
    parser.add_argument('--gamma', type=float, default=1)

    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1)
    parser.add_argument('--lr_decay3', type=float, default=1)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckps/target/')
    parser.add_argument('--folder', type=str, default='/home/jiujunhe/data/data_list/')
    parser.add_argument('--output_src', type=str, default='ckps/source/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
        args.net = 'resnet101'
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    folder = args.folder
    args.s_dset_path = folder + names[args.s] + '_list.txt'
    args.t_dset_path = folder + names[args.t] + '_list.txt'
    args.test_dset_path = folder + names[args.t] + '_list.txt'

    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, str(args.seed), names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.da, args.dset, str(args.seed), names[args.s][0].upper() + names[args.t][0].upper())
    args.name = names[args.s][0].upper() + names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'adv:{}_cls:{}_lr_:{}'.format(args.alpha, args.beta, args.lr)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)
    print('seed:%s' % args.seed)
