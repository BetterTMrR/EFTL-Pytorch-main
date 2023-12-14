# @Time    : 2021/1/28 18:56
# @FileName: utils.py
# @Software: PyCharm
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import math


class AvgLoss:
    def __init__(self):
        self.loss = 0.0
        self.n = 0

    def add_loss(self, loss):
        self.loss += loss
        self.n += 1

    def get_avg_loss(self):
        loss = self.loss / self.n
        self.loss = 0.0
        self.n = 0
        return loss


def test(F, C, L, cuda, note='accuracy on target domain is'):
    with torch.no_grad():
        F.eval()
        C.eval()
        correct_num = 0
        total_num = 0
        for idx, (imgs, labels) in enumerate(L):
            imgs, labels = imgs.cuda(cuda), labels.cuda(cuda)
            _, feats = F(imgs)
            output = C(feats)
            pred = output.argmax(dim=1)
            corrects = (pred == labels).sum()
            correct_num += int(corrects)
            total_num += imgs.size(0)
        print(note + ': %.2f%%' % (100*correct_num/total_num))
        return 100*correct_num/total_num


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, _



def comparsons(mu, sigma, distance, predictions, idx, T):
    prob = -(distance-(mu - sigma)*math.exp(-T))
    pred = (prob.argmax(dim=1)).unsqueeze(1)
    return predictions[idx, pred].squeeze()


def obtain_label(loader, netF, netB, netC, args, threshold=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(netB(feas))
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    prob, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if threshold:
        nms = torch.eye(args.class_num)[predict]
        threshold_ = (all_output * nms).sum(dim=0) / (1e-8 + nms.sum(dim=0))
        return threshold_.cuda()

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    log_str = ''
    k = args.class_num
    cls = torch.eye(args.class_num)
    for round in range(1):
        initc = all_output.t().mm(all_fea) / (all_output.sum(0).reshape(-1, 1) + 1e-8)
        distance = (1 - torch.cosine_similarity(all_fea.unsqueeze(1), initc.unsqueeze(0), dim=2)) / 2
        weighted_matrix = distance.clone()
        mu = ((weighted_matrix * all_output).sum(0) / (all_output.sum(0) + 1e-8)).view(-1, distance.size(1))
        sigma = (((weighted_matrix - mu).pow(2) * all_output).sum(0) / (all_output.sum(0) + 1e-8)).pow(0.5).view(-1,
                                                                                                                 distance.size(
                                                                                                                     1))
        topkvalue, predtopk = distance.topk(k, dim=1, largest=False)
        idx = torch.LongTensor([i for i in range(distance.size(0))]).unsqueeze(1)
        pred_label = distance.argmin(dim=1)
        mu = mu.repeat(all_fea.size(0), 1)[idx, predtopk]
        sigma = sigma.repeat(all_fea.size(0), 1)[idx, predtopk]
        distance = distance[idx, predtopk]
        pred_label2 = comparsons(mu, sigma, distance, predtopk, idx, args.T)
        all_output = cls[pred_label2]
        acc = np.sum(pred_label.float().numpy() == all_label.float().numpy()) / len(all_fea)
        log_str += 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100) + '   '
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str + '\n')

    return pred_label.cpu().numpy().astype('int')