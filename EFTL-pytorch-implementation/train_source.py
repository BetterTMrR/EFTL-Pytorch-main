import argparse
import os, sys
import os.path as osp
import numpy as np
import torch
import torch.optim as optim
from network import model_dict
from data_list import data_load
import random
from utils import AvgLoss, cal_acc, print_args, inv_lr_scheduler, CrossEntropyLabelSmooth


def train_source(args):
    avgloss = AvgLoss()
    dset_loaders = data_load(args)
    src_G, src_C = model_dict(args)
    param_group = []
    for k, v in src_G.named_parameters():
        if k[:4] == 'bott':
            param_group += [{'params': v, 'lr': 1}]
        else:
            param_group += [{'params': v, 'lr': 0.1}]

    for k, v in src_C.named_parameters():
        param_group += [{'params': v, 'lr': 1}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=4e-5, nesterov=True)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    acc_init = 0
    epoch_num = 0
    max_iter = args.max_epoch * len(dset_loaders["src_tr"])
    interval_iter = max_iter // 10
    iter_num = 0
    counter = 0
    src_C.train()
    src_G.train()
    SmoothLoss = CrossEntropyLabelSmooth(reduction=True, num_classes=args.class_num, epsilon=0.1)
    while iter_num < max_iter:
        try:
            inputs_source, labels_source, _ = iter_source.next()
        except:
            iter_source = iter(dset_loaders["src_tr"])
            inputs_source, labels_source, _ = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        if iter_num % len(dset_loaders["src_tr"]) == 0:
            epoch_num += 1
        iter_num += 1
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=args.lr)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        features = src_G(inputs_source)
        outputs_source = src_C(features)
        classifier_loss = SmoothLoss(outputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        avgloss.add_loss(classifier_loss.item())

        if epoch_num > args.max_epoch or counter > 3:
            break

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            src_G.eval()
            src_C.eval()
            avl = avgloss.get_avg_loss()
            acc_s_te, acc = cal_acc(dset_loaders['src_te'], src_G, src_C, args, False)
            log_str = 'Phase: Source Domain Pretraining, Task: {} ({}), Epoch: [{}/{}], Loss: {:.4f}, Source Acc.: {:.2f}%'.format(
                args.name_src, args.dset, epoch_num, args.max_epoch, avl, acc_s_te)
            args.out_file_src.write(log_str + '\n')
            args.out_file_src.flush()
            print(log_str + '\n')
            if acc_s_te >= 100.:
                counter += 1
            if acc_s_te > acc_init:
                acc_init = acc_s_te
                best_netG = src_G.state_dict()
                best_netF = src_C.state_dict()
            src_G.train()
            src_C.train()

    torch.save(best_netG,
               osp.join(args.output_dir_src, "source_{}_G.pt".format(args.seed)))
    torch.save(best_netF,
               osp.join(args.output_dir_src, "source_{}_F.pt".format(args.seed)))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Effective Target Labeling')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source domain")
    parser.add_argument('--t', type=int, default=1, help="target domain")
    parser.add_argument('--max_epoch', type=int, default=20, help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=48, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office_home',choices=['office', 'office_home', 'multi'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet34', help="vgg16, resnet34, alexnet")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--output', type=str, default='record')
    parser.add_argument('--da', type=str, default='semiDA', choices=['simiDA', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='full', choices=['full', 'val'])
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

    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper(), args.net)
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    if args.t != -1:
        _break = True
    else:
        _break = False
    if not osp.exists(
            osp.join(args.output_dir_src, "source_{}_G.pt".format(args.seed))):
        args.out_file_src = open(
            osp.join(args.output_dir_src, 'log_{}_source.txt'.format(args.seed)), 'w')
        args.out_file_src.write(print_args(args) + '\n')
        args.out_file_src.flush()
        print(osp.join(args.output_dir_src, "source_G.pt"))
        train_source(args)