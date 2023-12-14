from torch.autograd import Function
from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


res_dict = {"resnet18": models.resnet18, "resnet34":models.resnet34, "resnet50": models.resnet50,
"resnet101":models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class VGGBase(nn.Module):
    def __init__(self, pret=True, no_pool=False, bottleneck_dim=512):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pret)
        self.classifier = nn.Sequential(*list(vgg16.classifier.
                                              _modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features.
                                            _modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.bottleneck = nn.Linear(4096, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.bn = nn.BatchNorm1d(bottleneck_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        x = self.bn(self.bottleneck(x))
        return x


class ResBase(nn.Module):
    def __init__(self, res_name="resnet34", bottleneck_dim=512):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.bn = nn.BatchNorm1d(bottleneck_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x


class Classifier(nn.Module):
    def __init__(self, class_num, feature_dim=256):
        super(Classifier, self).__init__()
        self.type = type
        self.tmp = 0.05
        self.fc = nn.Linear(feature_dim, class_num)
        self.fc.apply(init_weights)

    def forward(self, x, reverse=False, alpha=1):
        if reverse:
            x = ReverseLayerF.apply(x, alpha)
        x = F.normalize(x, dim=1)
        x = self.fc(x) / self.tmp
        return x


def model_dict(args):
    if args.net == 'resnet34':
        G = ResBase(args.net, bottleneck_dim=256).cuda()
        F = Classifier(class_num=args.class_num, feature_dim=256).cuda()

    if args.net == 'resnet50':
        G = ResBase(args.net, bottleneck_dim=256).cuda()
        F = Classifier(class_num=args.class_num, feature_dim=256).cuda()

    if args.net == 'vgg':
        G = VGGBase(pret=True).cuda()
        F = Classifier(class_num=args.class_num, feature_dim=512).cuda()

    return G, F