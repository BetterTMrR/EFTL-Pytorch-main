import argparse
import os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from network import model_dict
from data_list import data_load_target, ImageList_idx, image_train
import random
import time
from utils import AvgLoss, cal_acc, print_args, inv_lr_scheduler, \
    pseudo_labeling_CLOSE, active_strategy, CrossEntropyLabelSmooth


def ssl_training(args):
    avg_los = AvgLoss()
    dset_loaders = data_load_target(args)
    ssl_G, ssl_C = model_dict(args)
    ssl_G.load_state_dict(torch.load(
        osp.join(args.output_dir_src, "source_{}_G.pt".format(args.seed))))
    ssl_C.load_state_dict(torch.load(
        osp.join(args.output_dir_src, "source_{}_F.pt".format(args.seed))))
    ssl_G.eval()
    ssl_C.eval()
    imgs, labels = pseudo_labeling_CLOSE(dset_loaders, ssl_G, ssl_C, args)
    txt_lbd = [img + ' ' + label for img, label in zip(imgs, labels)]
    dataset_lbd = ImageList_idx(txt_lbd, root=args.root, transform=image_train(alexnet=args.net == 'alexnet'))
    loader_lbd = DataLoader(dataset_lbd, batch_size=args.batch_size // 2, shuffle=True,
                            num_workers=args.worker,
                            drop_last=False)

    dset_loaders['tar_lbd'] = loader_lbd

    ssl_G.train()
    ssl_C.train()

    param_group_ssl = []
    param_group_ssl_f = []

    for k, v in ssl_G.named_parameters():
        if k[:4] == 'bott':
            param_group_ssl += [{'params': v, 'lr': 1}]
        else:
            param_group_ssl += [{'params': v, 'lr': 0.1}]

    for k, v in ssl_C.named_parameters():
        param_group_ssl_f += [{'params': v, 'lr': 1}]

    optimizer_ssl = optim.SGD(param_group_ssl, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_ssl_f = optim.SGD(param_group_ssl_f, momentum=0.9, weight_decay=0.0005, nesterov=True)

    param_lr_g = []
    for param_group in optimizer_ssl.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_ssl_f.param_groups:
        param_lr_f.append(param_group["lr"])

    budget = args.shot * args.class_num
    active_num = [budget // args.rounds] * (args.rounds - 1)
    active_num.extend([budget - sum(active_num)])
    active_epoch = [(2 * (i + 1) - args.start) for i in range(len(active_num))]
    print(active_num)
    print(active_epoch)

    acc_best = 0
    epoch_num = 0
    interval_iter = len(dset_loaders["tar_un"])
    iter_num = 0
    max_iter = max(50000, args.max_epoch * interval_iter)
    SmoothLoss = CrossEntropyLabelSmooth(reduction=False, num_classes=args.class_num, epsilon=args.epsilon)

    active_idx = 0
    loss_list = []
    acc_list = []
    start_time = time.time()
    while iter_num < max_iter:
        try:
            inputs_tgt_un, inputs_tgt_un_trans, _, tarIdx = iter_tgt_un.next()
        except:
            iter_tgt_un = iter(dset_loaders["tar_un"])
            inputs_tgt_un, inputs_tgt_un_trans, _, tarIdx = iter_tgt_un.next()
        try:
            inputs_tgt, labels_tgt, _tarIdx = iter_tgt.next()
        except:
            iter_tgt = iter(dset_loaders["tar_lbd"])
            inputs_tgt, labels_tgt, _tarIdx = iter_tgt.next()

        if args.use_src:
            try:
                inputs_src, labels_src, _srcIdx = iter_source.next()
            except:
                iter_source = iter(dset_loaders["src_tr"])
                inputs_src, labels_src, _srcIdx = iter_source.next()
        else:
            labels_src = []

        if inputs_tgt_un.size(0) == 1 or inputs_tgt.size(0) == 1:
            continue

        optimizer_ssl = inv_lr_scheduler(param_lr_g, optimizer_ssl, iter_num, init_lr=args.lr)
        optimizer_ssl_f = inv_lr_scheduler(param_lr_f, optimizer_ssl_f, iter_num, init_lr=args.lr)

        if args.use_src:
            inputs_lbd = torch.cat([inputs_src, inputs_tgt], dim=0).cuda()
            labels_lbd = torch.cat([labels_src, labels_tgt], dim=0).cuda()
        else:
            inputs_lbd = inputs_tgt.cuda()
            labels_lbd = labels_tgt.cuda()

        outputs_lbd = ssl_C(ssl_G(inputs_lbd).cuda())
        classifier_loss_ssl = SmoothLoss(outputs_lbd, labels_lbd, len(labels_src)).mean()
        optimizer_ssl.zero_grad()
        optimizer_ssl_f.zero_grad()
        classifier_loss_ssl.backward()
        optimizer_ssl.step()
        optimizer_ssl_f.step()

        if args.method == 'FixMatch':
            inputs_un = torch.cat([inputs_tgt_un.cuda(), inputs_tgt_un_trans.cuda()], dim=0)
            features = ssl_G(inputs_un)
            features, features_trans = features.chunk(dim=0, chunks=2)
            output_trans_tar = ssl_C(features_trans)

            output_tar = ssl_C(features)
            classifier_tar = nn.Softmax(dim=1)(output_tar)
            prob_tar, pred_tar = classifier_tar.max(1)
            mask_tar = prob_tar.ge(args.th).float()

            transfer_loss = (SmoothLoss(output_trans_tar, pred_tar) * mask_tar).mean()

        elif args.method == 'MME':
            features = ssl_G(inputs_tgt_un.cuda())
            output_tar = ssl_C(features, reverse=True, alpha=1)
            output_tar_softmax = nn.Softmax(dim=1)(output_tar)
            transfer_loss = args.lam * (output_tar_softmax * torch.log(output_tar_softmax + 1e-8)).sum(1).mean()

        elif args.method == 'FixMME':
            features = ssl_G(inputs_tgt_un.cuda())
            output_tar = ssl_C(features, reverse=True, alpha=1)
            output_tar_softmax = nn.Softmax(dim=1)(output_tar)
            mme_loss = args.lam * (output_tar_softmax * torch.log(output_tar_softmax + 1e-8)).sum(1).mean()

            inputs_un = torch.cat([inputs_tgt_un.cuda(), inputs_tgt_un_trans.cuda()], dim=0)
            features = ssl_G(inputs_un)
            features, features_trans = features.chunk(dim=0, chunks=2)
            output_trans_tar = ssl_C(features_trans)

            output_tar = ssl_C(features)
            classifier_tar = nn.Softmax(dim=1)(output_tar)
            prob_tar, pred_tar = classifier_tar.max(1)
            mask_tar = prob_tar.ge(args.th).float()

            transfer_loss = (SmoothLoss(output_trans_tar, pred_tar) * mask_tar).mean() + mme_loss

        total_loss = transfer_loss

        optimizer_ssl.zero_grad()
        optimizer_ssl_f.zero_grad()
        total_loss.backward()
        optimizer_ssl.step()
        optimizer_ssl_f.step()

        avg_los.add_loss(total_loss.item())

        if iter_num % interval_iter == 0:
            if epoch_num in active_epoch:
                active_strategy(dset_loaders, ssl_G, ssl_C, active_num[active_idx], args)
                active_idx += 1
            epoch_num += 1

        iter_num += 1

        if iter_num % (args.interval * interval_iter) == 0 or iter_num == max_iter:
            ssl_G.eval()
            ssl_C.eval()
            avg_losses = avg_los.get_avg_loss()
            end_time = time.time()

            time_need = (end_time - start_time) * ((args.max_epoch - epoch_num) / args.interval)
            time_need = '{}h{}m{}s'.format(int(time_need // 3600), int((time_need % 3600) // 60), int(time_need % 60))
            time_cost = '{}m{}s. Time need: {}.'.format(int((end_time - start_time) // 60),
                                                                    int((end_time - start_time) % 60), time_need)
            test_acc, acc = cal_acc(dset_loaders['tar_test'], ssl_G, ssl_C, args, args.dset=='VISDA-C', True)
            if test_acc >= acc_best:
                acc_best = test_acc
            acc_list.append(test_acc)
            loss_list.append(avg_losses)
            log_str = 'Phase: Adaptation Training. Method: {}. Task: {} ({}). Epoch: [{}/{}]. Acc: ({:.1f}%) [({:.1f}%) (best)]. Test num: {}. Time: {}'.format(
                args.method, args.name, args.dset, epoch_num, args.max_epoch, test_acc, acc_best, len(dset_loaders['tar_test'].dataset), time_cost)

            args.out_file_ssl.write(log_str + '\n')
            args.out_file_ssl.flush()
            print(log_str + '\n')
            ssl_G.train()
            ssl_C.train()
            start_time = time.time()
        if epoch_num > args.max_epoch:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Effective Target Labeling')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source domain")
    parser.add_argument('--t', type=int, default=1, help="target domain")
    parser.add_argument('--hop', type=int, default=2, help="2-order neighbours")
    parser.add_argument('--rounds', type=int, default=6, help="rounds of active strategy")
    parser.add_argument('--max_epoch', type=int, default=100, help="training epochs")

    parser.add_argument('--k1', type=int, default=3, help="M1 // 3")
    parser.add_argument('--start', type=int, default=0, help="start to NODE")
    parser.add_argument('--interval', type=int, default=5, help="print interval")
    parser.add_argument('--use_src', action="store_true", help="use source data")
    parser.add_argument('--lam', type=float, default=.1, help="lambda for MME")
    parser.add_argument('--alpha', type=float, default=0.1, help="NODE-MARGIN trade-off")
    parser.add_argument('--epsilon', type=float, default=0.1, help="label smoothing parameter")
    parser.add_argument('--th', type=float, default=0.85, help="hyper-parameter of threshold")
    parser.add_argument('--batch_size', type=int, default=48, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office_home', choices=['office', 'office_home', 'multi'], help='data set')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet34', help="vgg16, resnet34, alexnet")
    parser.add_argument('--method', type=str, default='FixMME', help='SSDA method')
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--shot', type=int, default=3)
    parser.add_argument('--output', type=str, default='record')
    parser.add_argument('--da', type=str, default='semiDA')
    args = parser.parse_args()
    torch.autograd.set_detect_anomaly(True)
    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'Real']
        args.names = ['Art', 'Clipart', 'Product', 'Real']
        args.root = './data/'  # root for images
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.names = ['amazon', 'dslr', 'webcam']
        args.root = './data/'  # root for images
        args.class_num = 31
    if args.dset == 'multi':
        names = ['clipart', 'painting', 'real', 'sketch']
        args.names = ['clipart', 'painting', 'real', 'sketch']
        args.class_num = 126
        args.root = './data/'  # root for images

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    alexnet = False
    if args.net == 'alexnet':
        print('reset batch size!')
        alexnet = True
        args.batch_size = 64
    folder = './data/source_txt/{}/'.format(args.dset)
    args.s_dset_path = folder + "labeled_images_{}.txt".format(names[args.s])
    args.t_dset_path = folder + "labeled_images_{}.txt".format(names[args.t])
    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    args.output_dir_tgt = osp.join(args.output, args.da, args.dset, args.name, args.net)
    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper(), args.net)
    if not osp.exists(args.output_dir_tgt):
        os.system('mkdir -p ' + args.output_dir_tgt)
    if not osp.exists(args.output_dir_tgt):
        os.mkdir(args.output_dir_tgt)

    log_str = print_args(args)
    meth_param = {'Fixmatch': 'th:{}'.format(args.th),
                  'MME': 'lam:{}'.format(args.lam),
                  'FixMME': 'th:{},lam:{}'.format(args.th, args.lam)}
    out_file_ssl_path = osp.join(args.output_dir_tgt,
                                      '{}({})_shot:{}_alpha:{}_{}.txt'.format(
                                          args.method,
                                          meth_param[args.method],
                                          args.shot,
                                          args.alpha,
                                          args.seed
                                      ))
    args.out_file_ssl = open(out_file_ssl_path, 'w')

    args.out_file_ssl.write(log_str + '\n')
    args.out_file_ssl.flush()
    ssl_training(args)