import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)


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
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, use_gpu=True, reduction=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, weights):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        loss = (- targets * log_probs).sum(dim=1)
        loss = loss * weights.cuda()
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


class ClusteringDistribution(nn.Module):
    def __init__(self):
        super(ClusteringDistribution, self).__init__()

    def forward(self, features, prototypes):
        # features = features.cuda()
        # prototypes = prototypes.cuda()
        norm_list = []
        for feature in features:
            feature = feature.expand_as(prototypes)
            norm_dis = torch.norm(feature-prototypes, p=2, dim=1)
            norm_dis = torch.pow(1 + norm_dis, -1)
            norm_list.append(norm_dis)
        distribution_ = torch.stack(norm_list, dim=0)
        distribution_ = nn.Softmax(dim=1)(distribution_)

        return distribution_


class LabelEmbedding(nn.Module):
    def __init__(self, num_class, embedding_matrix):
        super(LabelEmbedding, self).__init__()
        self.classes = num_class
        self.EmbeddingLayer = embedding_matrix

    def forward(self, x, label, weight):
        # print(label)
        # print(self.EmbeddingLayer.size())
        dense_label = self.EmbeddingLayer[label]
        # print(dense_label.size())
        # print(x.size())
        # x = F.normalize(x, dim=1)
        # loss = torch.sum(dense_label * torch.log(x + 1e-6), dim=1) * weight
        loss = torch.cosine_similarity(dense_label, x, dim=1) * weight
        return -loss.mean()


def TarDisClusterLoss(output, target=None):
    prob_p = F.softmax(output, dim=1)
    prob_q2 = prob_p / prob_p.sum(0, keepdim=True).pow(0.5)
    prob_q2 /= prob_q2.sum(1, keepdim=True)
    prob_q = prob_q2
    if target is not None:
        loss = - (prob_q * F.log_softmax(target.detach(), dim=1)).sum(1).mean()
        # loss = (prob_q * torch.log(1 - nn.Softmax(dim=1)(target))).sum(1).mean()
    else:
        loss = - (prob_q * F.log_softmax(output, dim=1)).sum(1).mean()
    return loss


def adentropy(target_data, input_data):
    target_data_ = target_data.argmax(dim=1)
    target_data_ = torch.zeros(input_data.size()).scatter_(1, target_data_.unsqueeze(1).cpu(), 1).cuda()
    input_data = nn.Softmax(dim=1)(input_data)
    # entropy = (nn.Softmax(dim=1)(target_data) * torch.log(1 - input_data + 1e-6)).sum(dim=1)
    entropy = (target_data_ * torch.log(1 - input_data + 1e-6)).sum(dim=1)
    return entropy.mean()


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.use_gpu = True
        self.reduction = True

    def forward(self, input_f, input_f_prime):
        targets = input_f.argmax(dim=1)
        probs = nn.Softmax(dim=1)(input_f_prime)
        targets = torch.zeros(probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        # targets = nn.Softmax(dim=1)(input_f_prime)
        if self.use_gpu:
            targets = targets.cuda()
        loss = (torch.log(1 - probs*targets + 1e-6)).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss




class MAE(nn.Module):
    def __init__(self, num_classes, use_gpu=True, reduction=True):
        super(MAE, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        prob = nn.Softmax(dim=1)(inputs)
        targets = torch.zeros(prob.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        loss = torch.abs(targets - prob).mean(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


def topk_loss(inputs_f, inputs, pred, thres):
    inputs_ = nn.Softmax(dim=1)(inputs_f)
    prob, _ = torch.max(inputs_, dim=1)
    idx = torch.where(prob > thres[pred])[0]
    if len(idx) > 0:
        inputs = inputs[idx]
        labels = pred[idx]
        loss = nn.CrossEntropyLoss()(inputs, labels)
    else:
        loss = torch.Tensor([0])
    return loss






