import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from randaugment import RandAugmentMC


def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, root=None, labels=None, transform=None, strong_transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.samples = np.array(imgs)
        self.imgs, self.labels = self.samples.T
        self.transform = transform
        self.target_transform = target_transform
        self.strong_transform = strong_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index], self.labels[index]
        img_o = self.loader(self.root + path)
        if self.transform is not None:
            img = self.transform(img_o)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.strong_transform is not None:
            img_trans = self.strong_transform(img_o)
            return img, img_trans, int(target)

        return img, int(target)

    def __len__(self):
        return len(self.imgs)

    def add_item(self, imgs, labels):
        self.imgs = np.concatenate((self.imgs, imgs), axis=0)
        self.labels = np.concatenate((self.labels, labels), axis=0)
        return self.imgs, self.labels

    def remove_item(self, reduced):
        reduced = reduced.astype('int64')
        self.imgs = np.delete(self.imgs, reduced, axis=0)
        self.labels = np.delete(self.labels, reduced, axis=0)
        return self.imgs, self.labels


class ImageList_idx(Dataset):
    def __init__(self, image_list, root, labels=None, transform=None, strong_transform=None, target_transform=None, mode='RGB'):
        self.image_list = image_list
        imgs = make_dataset(image_list, labels)
        self.root = root
        self.samples = np.array(imgs)
        self.imgs, self.labels = self.samples.T
        self.exist = np.array([i for i in range(len(self.imgs))], dtype='int64')
        self.mapping_idx2item = {i: self.exist[i] for i in range(len(self.imgs))}
        self.mapping_item2idx = {self.exist[i]: i for i in range(len(self.imgs))}
        self.global_marker = np.array([1 for i in range(len(self.imgs))])
        self.transform = transform
        self.n_true = 0
        self.target_transform = target_transform
        self.strong_transform = strong_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index], self.labels[index]
        img_o = self.loader(self.root+path)
        if self.transform is not None:
            img = self.transform(img_o)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.strong_transform is not None:
            img_trans = self.strong_transform(img_o)
            return img, img_trans, int(target), index

        return img, int(target), index

    def __len__(self):
        return len(self.imgs)

    def add_item(self, imgs, labels):
        self.imgs = np.concatenate((self.imgs, imgs), axis=0)
        self.labels = np.concatenate((self.labels, labels), axis=0)
        return self.imgs, self.labels

    def remove_item(self, reduced):
        reduced = reduced.astype('int64')
        self.imgs = np.delete(self.imgs, reduced, axis=0)
        self.labels = np.delete(self.labels, reduced, axis=0)
        self.exist = np.delete(self.exist, reduced, axis=0)
        self.global_marker = np.delete(self.global_marker, reduced, axis=0)
        self.mapping_idx2item = {i: self.exist[i] for i in range(len(self.imgs))}
        self.mapping_item2idx = {self.exist[i]: i for i in range(len(self.imgs))}
        assert len(self.imgs) == len(self.labels)
        return self.imgs, self.labels

    def add_pseudo_labeled_examples(self, imgs, labels):
        self.imgs = np.concatenate([self.imgs, imgs], axis=0)
        self.labels = np.concatenate([self.labels, labels], axis=0)

    def remove_pseudo_labeled_examples(self):
        self.imgs = self.imgs[:self.n_true]
        self.labels = self.labels[:self.n_true]


def image_strong(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if alexnet:
        crop_size = 227
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        RandAugmentMC(n=2, m=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    if alexnet:
        crop_size = 227
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if alexnet:
        crop_size = 227
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
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()

    dsets["src_tr"] = ImageList_idx(txt_src, root=args.root, transform=image_train(alexnet=args.net=='alexnet'))
    dsets["tar_tr"] = ImageList_idx(txt_tar, root=args.root, transform=image_train(alexnet=args.net == 'alexnet'))

    dsets["src_te"] = ImageList_idx(txt_src, root=args.root, transform=image_test(alexnet=args.net=='alexnet'))
    dsets["tar_te"] = ImageList_idx(txt_tar, root=args.root, transform=image_test(alexnet=args.net=='alexnet'))

    dset_loaders["src_tr"] = DataLoader(dsets["src_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)

    dset_loaders["src_te"] = DataLoader(dsets["src_te"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                        drop_last=False)

    dset_loaders["tar_tr"] = DataLoader(dsets["tar_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)

    dset_loaders["tar_te"] = DataLoader(dsets["tar_te"], batch_size=train_bs * 2, shuffle=True, num_workers=args.worker,
                                          drop_last=False)
    return dset_loaders


def data_load_target(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_src = open(args.s_dset_path).readlines()

    dsets["tar_un"] = ImageList_idx(txt_tar, root=args.root, transform=image_train(alexnet=args.net=='alexnet'), strong_transform=image_strong(alexnet=args.net=='alexnet'))
    dsets["tar_test"] = ImageList_idx(txt_tar, root=args.root, transform=image_test(alexnet=args.net=='alexnet'))
    dsets["src_tr"] = ImageList_idx(txt_src, root=args.root, transform=image_train(alexnet=args.net=='alexnet'))

    dset_loaders["src_tr"] = DataLoader(dsets["src_tr"], batch_size=train_bs // 2, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["tar_un"] = DataLoader(dsets["tar_un"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                           drop_last=False)

    dset_loaders["tar_test"] = DataLoader(dsets["tar_test"], batch_size=train_bs * 3, shuffle=True, num_workers=args.worker,
                                        drop_last=False)

    return dset_loaders

def data_load_base(args):
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar_lbd = open(args.t_dset_path_lbd).readlines()
    txt_tar_unlbd = open(args.t_dset_path_unlbd).readlines()
    txt_src = open(args.s_dset_path).readlines()

    dsets["tar_un"] = ImageList_idx(txt_tar_unlbd, root=args.root, transform=image_train(alexnet=args.net=='alexnet'), strong_transform=image_strong(alexnet=args.net=='alexnet'))
    dsets["tar_test"] = ImageList_idx(txt_tar_unlbd, root=args.root, transform=image_test(alexnet=args.net=='alexnet'))
    dsets["src_tr"] = ImageList_idx(txt_src, root=args.root, transform=image_train(alexnet=args.net=='alexnet'))
    dsets["tar_tr"] = ImageList_idx(txt_tar_lbd, root=args.root, transform=image_train(alexnet=args.net=='alexnet'))

    dset_loaders["src_tr"] = DataLoader(dsets["src_tr"], batch_size=train_bs // 2, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["tar_lbd"] = DataLoader(dsets["tar_tr"], batch_size=train_bs // 2, shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["tar_un"] = DataLoader(dsets["tar_un"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                           drop_last=False)

    dset_loaders["tar_test"] = DataLoader(dsets["tar_test"], batch_size=train_bs * 3, shuffle=True, num_workers=args.worker,
                                        drop_last=False)

    print('label data: {} (source, {}); {} (target, ({})). unlabeled target data: {},({})'.format(
        len(dset_loaders["src_tr"]),
        len(txt_src),
        len(dset_loaders["tar_lbd"]),
        len(txt_tar_lbd),
        len(dset_loaders["tar_test"]),
        len(txt_tar_unlbd)
    ))

    return dset_loaders