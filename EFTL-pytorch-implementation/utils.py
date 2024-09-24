import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader
import math
from collections import Counter
from data_list import ImageList_idx, image_train


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma=0.0001,
                     power=0.75, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (- power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1
    return optimizer


class AvgLoss:
    def __init__(self):
        self.loss = 0.0
        self.n = 1e-8

    def add_loss(self, loss):
        self.loss += loss
        self.n += 1

    def get_avg_loss(self):
        avg_loss = self.loss / self.n
        self.n = 1e-8
        self.loss = 0.0
        return avg_loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets, n_src=0):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        n_total = inputs.shape[0]
        epsilon = torch.cat([torch.ones((n_src, 1)), torch.zeros(n_total-n_src, 1)], dim=0).cuda().float()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        epsilon = self.epsilon * epsilon if n_src > 0 else self.epsilon
        targets = (1 - epsilon) * targets + epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss


def cal_acc(loader, G, F, args, flag=False, details=False, save_name=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = G(inputs)
            outputs = F(fea)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_fea = fea.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_fea = torch.cat((all_fea, fea.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    prob, predict = torch.max(all_output, 1)

    if details:
        n = all_output.size(0)
        class_balance_dict = {}
        truth_balance = [_ for _ in all_label.numpy().astype('int').squeeze()]
        predict_ = [_ for _ in predict.numpy().astype('int').squeeze()]
        class_balance = all_output.numpy().mean(0)
        gt = Counter(truth_balance)
        pred_balance = Counter(predict_)
        for cls in range(all_output.size(1)):
            class_balance_dict[cls] = [gt[cls] / n, pred_balance[cls] / n, class_balance[cls]]

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if save_name is not None:
        np.save(args.output_dir_tgt + '/{}_{}_output.npy'.format(save_name, args.method), all_output.cpu().numpy())
        np.save(args.output_dir_tgt + '/{}_{}_fea.npy'.format(save_name, args.method), all_fea.cpu().numpy())
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc, accuracy
    else:
        if details:
            return round(accuracy * 100, 2), all_fea.numpy()
        else:
            return round(accuracy * 100, 2), 0


def adjacent_matrix(features, k, M):
    features = F.normalize(features, 1)
    global_idx = torch.LongTensor([i for i in range(features.size(0))])
    cosine_simi = features.mm(features.t())
    _, topk_idx = cosine_simi.topk(dim=1, k=k)
    topk_idx = topk_idx[:, 1:]
    connected = (topk_idx[:, :M][topk_idx] == global_idx.unsqueeze(-1).unsqueeze(-1)).sum(2)
    edges = torch.where(connected > 0, torch.ones(connected.size()), torch.zeros(connected.size()))
    return topk_idx, edges


def pseudo_labeling_CLOSE(loader, G, F, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["tar_test"])
        for i in range(len(loader["tar_test"])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[-2]
            idx = data[-1]
            inputs = inputs.cuda()
            fea = G(inputs)
            out = F(fea)
            if start_test:
                all_label = labels.float().cpu()
                all_fea = fea.float().cpu()
                all_out = out.float().cpu()
                all_idx = idx.float().cpu()
                start_test = False
            else:
                all_label = torch.cat((all_label, labels.float().cpu()), 0)
                all_out = torch.cat((all_out, out.float().cpu()), 0)
                all_fea = torch.cat((all_fea, fea.float().cpu()), 0)
                all_idx = torch.cat((all_idx, idx.float().cpu()), 0)

    all_idx = all_idx.long()
    samples = loader["tar_test"].dataset.imgs, loader["tar_test"].dataset.labels
    all_out = nn.Softmax(dim=-1)(all_out)
    prob, pred = all_out.max(1)
    acc1 = (pred.float() == all_label.float()).float().mean().numpy()
    M1 = (len(all_label) // (3 * all_out.shape[1])) // 5
    kk = max(M1, 5)
    pred1, prob1 = kmeans_clustering(all_fea, all_out, all_label)
    prob2, pred2 = gmm_clustering(all_out, all_fea, all_label)

    idx1 = pl(prob1, k=kk, total_lb=all_label)
    idx2 = pl(prob2, k=kk, total_lb=all_label)

    idxes = []
    pseudo_labels = []

    for ii, (i1, i2) in enumerate(zip(idx2.t(), idx1.t())):
        consistency = (i1.unsqueeze(0) == i2.unsqueeze(1)).sum(1)
        idxes.extend(list(i1[consistency > 0].numpy()))
        pseudo_labels.extend(list(ii * torch.ones(i1[consistency > 0].shape).numpy().astype('int64')))
        args.out_file_ssl.write(f'class:{ii}' + ' ' + str(len(i1[consistency > 0])) + ' ' + str((all_out.sum(0)[ii] / all_out.sum(0).max()).cpu().numpy()) + ' ' +
              str((ii * torch.ones(i1[consistency > 0].shape) == all_label[i1[consistency > 0]]).float().mean().numpy()) + '\n'
              )
        args.out_file_ssl.flush()

    confidence_idxes = []

    idxes = torch.LongTensor(idxes)
    pseudo_labels = torch.LongTensor(pseudo_labels)
    acc2 = (pseudo_labels.cpu().float() == all_label[idxes]).float().mean().numpy()
    log_str = 'Source Model Accuracy: {:.2f}%, Num. of Pseudo-labels: {}, Num. of Unlabeled Samples: {}, Pseudo-label Accuracy: {:.2f}%'.format(acc1*100, len(idxes)-len(confidence_idxes), len(all_out), acc2*100)
    print(log_str)
    args.out_file_ssl.write(log_str + '\n')
    args.out_file_ssl.flush()
    loader["tar_un"].dataset.remove_item(all_idx[idxes].numpy())
    return samples[0][all_idx[idxes].numpy()], pseudo_labels.numpy().astype('int64').astype('str')


def kmeans_clustering(features, outputs, labels, k=3, hard=False):
    '''
    :param features: [n, m]
    :param outputs: [n, k]
    :param labels: [n, ]
    :param k: int
    :return:
    '''
    pred = outputs.argmax(1)
    acc = (pred.squeeze()==labels.squeeze()).sum().numpy() / len(labels)
    if hard:
        onehot = torch.eye(outputs.shape[1])
        outputs = onehot[pred]
    features = torch.nn.functional.normalize(features, dim=1)
    initc = outputs.t().mm(features)
    initc /= outputs.sum(0).reshape(-1, 1) + 1e-8
    log = ''
    for i in range(k):
        # simi = features.mm(initc.t())
        simi = torch.cosine_similarity(features.unsqueeze(1), initc.unsqueeze(0), dim=2)
        pred = simi.argmax(1)
        onehot = torch.eye(outputs.shape[1])
        outputs = onehot[pred]
        initc = outputs.t().mm(features)
        initc /= outputs.sum(0).reshape(-1, 1) + 1e-8
        acc1 = (pred.squeeze() == labels.squeeze()).sum().numpy() / len(labels)

        log += 'Accuracy= {:.2f}% -> {:.2f}%   '.format(acc*100, acc1*100)
    # print(log)
    return pred, torch.nn.Softmax(dim=-1)(simi / 0.05)


def gmm(all_fea, pi, mu, all_output):
    Cov = []
    dist = []
    log_probs = []
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:, i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + 1e-6 * torch.eye(
            temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += 1e-6 * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5 * (Covi.shape[0] * np.log(2 * math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)

    return zz, gamma


def gmm_clustering(all_output, all_fea, all_label):
    all_output = all_output.cuda()
    all_fea = all_fea.cuda()
    all_label = all_label.cuda()
    _, predict = torch.max(all_output, 1)

    all_output_logit = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + 1e-2), dim=1)
    unknown_weight = 1 - ent / np.log(all_output.shape[1])

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if args.distance == 'cosine':
    # all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = torch.nn.functional.normalize(all_fea, dim=1)

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    initc = torch.matmul(aff.t(), (all_fea))
    initc = initc / (1e-8 + aff.sum(dim=0)[:, None])

    uniform = torch.ones(len(all_fea), all_output.shape[1]) / all_output.shape[1]
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm((all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)
    acc = (pred_label == all_label).float().mean().cpu().numpy()
    log_str = 'Model Prediction: Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100) + '\n'

    # print(log_str)
    return gamma.cpu(), pred_label.cpu()


def pl(outputs, total_lb, k=5):
    prob, idx = outputs.topk(k, dim=0)
    pred_labels = torch.LongTensor([i for i in range(outputs.shape[1])]).view(1, -1).repeat(k, 1).view(-1)
    gt = total_lb[idx.view(-1)]
    # print((gt.cpu().float()==pred_labels.float()).float().mean())
    return idx


def active_strategy(loader, G, F, active_num, args):
    G.eval()
    F.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["tar_un"])
        for i in range(len(loader["tar_un"])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[-2]
            idx = data[-1]
            inputs = inputs.cuda()
            fea = G(inputs)
            out = F(fea)
            if start_test:
                all_label = labels.float().cpu()
                all_fea = fea.float().cpu()
                all_out = out.float().cpu()
                all_idx = idx.float().cpu()
                start_test = False
            else:
                all_label = torch.cat((all_label, labels.float().cpu()), 0)
                all_out = torch.cat((all_out, out.float().cpu()), 0)
                all_fea = torch.cat((all_fea, fea.float().cpu()), 0)
                all_idx = torch.cat((all_idx, idx.float().cpu()), 0)

    all_idx = all_idx.long()
    all_out = nn.Softmax(dim=-1)(all_out)
    prob_sort, _ = all_out.sort(dim=1, descending=True)
    global_marker = torch.from_numpy(loader["tar_un"].dataset.global_marker)[all_idx]
    M1 = max(len(all_idx) // (3 * all_out.shape[1]), 10)
    M2 = M1 // 5
    nearest_idxes, edges = adjacent_matrix(all_fea, k=M1, M=M2)
    chosen = []
    degree = edges.sum(1)
    degree /= (degree.max() + 1e-8)
    margin = prob_sort[:, 0] - prob_sort[:, 1]
    margin = (1 - margin) / (1 - margin).max()
    uncertainty = args.alpha * degree + margin
    uncertainty[global_marker!=1] = -1
    for _ in range(active_num):
        _, sorted_idxes = uncertainty.sort(descending=True)
        idx_chosen = sorted_idxes[0]
        chosen.append(idx_chosen)
        uncertainty[idx_chosen] = -1
        for __ in range(args.hop):
            connected_idx = nearest_idxes[idx_chosen][edges[idx_chosen] > 0]
            connected_idx = torch.from_numpy(np.unique(connected_idx.numpy())).long()
            uncertainty[connected_idx] = -1
            idx_chosen = connected_idx
    chosen = torch.LongTensor(chosen)
    mark_idxes = chosen.clone()

    # mask the 1-order neighbors of queried samples
    for _ in range(1):
        mark_idxes = nearest_idxes[mark_idxes][edges[mark_idxes] > 0]
        mark_idxes = torch.from_numpy(np.unique(mark_idxes.numpy())).long()
    mark_idxes = np.unique(mark_idxes.numpy()).astype('int64')
    samples = loader["tar_un"].dataset.imgs, loader["tar_un"].dataset.labels

    remove_item = [loader["tar_un"].dataset.mapping_idx2item[i] for i in all_idx[chosen].numpy()]
    remove_idx = np.array([loader["tar_test"].dataset.mapping_item2idx[item] for item in remove_item])

    loader["tar_un"].dataset.global_marker[all_idx.numpy()[mark_idxes]] = -1
    loader["tar_un"].dataset.remove_item(all_idx[chosen].numpy())
    loader["tar_test"].dataset.remove_item(remove_idx)

    try:
        loader['tar_lbd'].dataset.add_item(samples[0][all_idx[chosen].numpy()], samples[1][all_idx[chosen].numpy()])
    except:
        print('There are no labeled data in the target dataset.')
        txt_lbd = [img + ' ' + label for img, label in zip(samples[0][all_idx[chosen].numpy()], samples[1][all_idx[chosen].numpy()])]
        dataset_lbd = ImageList_idx(txt_lbd, root=args.root, transform=image_train(alexnet=args.net == 'alexnet'))
        loader_lbd = DataLoader(dataset_lbd, batch_size=args.batch_size // 2, shuffle=True,
                                num_workers=args.worker,
                                drop_last=False)
        loader['tar_lbd'] = loader_lbd

    lbd_items = samples[0][all_idx[chosen].numpy()]
    test_items = loader["tar_test"].dataset.imgs
    len_marked = loader["tar_un"].dataset.global_marker[loader["tar_un"].dataset.global_marker!=1].shape[0]
    len_total = len(loader["tar_un"].dataset.global_marker)

    print('Select {} instances. Totally mark [{}/{}] instances. Alpha: {:.2f}.'.format(len(chosen), len_marked, len_total, args.alpha))
    args.out_file_ssl.write('Select {} instances. Totally mark [{}/{}] instances. Alpha: {:.2f}.'.format(len(chosen), len_marked, len_total, args.alpha) + '\n'
        )

    # check whether there exist labeled samples in the test set
    num = 0
    for item in lbd_items:
        num += (item.reshape(-1, 1) == test_items.reshape(-1, 1)).sum()

    print('There are {} labeled instances in the test set.'.format(num))

    args.out_file_ssl.flush()
    G.train()
    F.train()