""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

# from torch.optim.lr_scheduler import _LRScheduler
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_network(args, img_channels=3, num_classes=1000):
    """ return given network
    """
    if args == 'resnet50':
        from models.ResNet import ResNet50
        net = ResNet50(img_channels, num_classes)
    elif args == 'resnet101':
        from models.ResNet import ResNet101
        net = ResNet101(img_channels, num_classes)
    elif args == 'resnet152':
        from models.ResNet import ResNet152
        net = ResNet152(img_channels, num_classes)
    elif args == 'resnet18':
        from models.ResNet18 import ResNet18
        net = ResNet18()
    elif args == 'resnext26_2x64d':
        from models.ResNext import resnext26_2x64d
        net = resnext26_2x64d(num_classes=num_classes)
    elif args == 'resnext26_4x32d':
        from models.ResNext import resnext26_4x32d
        net = resnext26_4x32d(num_classes=num_classes)
    elif args == 'resnext26_8x16d':
        from models.ResNext import resnext26_8x16d
        net = resnext26_8x16d(num_classes=num_classes)
    elif args == 'resnext26_16x8d':
        from models.ResNext import resnext26_16x8d
        net = resnext26_16x8d(num_classes=num_classes)
    elif args == 'resnext26_32x4d':
        from models.ResNext import resnext26_32x4d
        net = resnext26_32x4d(num_classes=num_classes)
    elif args == 'resnext26_64x2d':
        from models.ResNext import resnext26_64x2d
        net = resnext26_64x2d(num_classes=num_classes)
    elif args == 'resnext50_2x64d':
        from models.ResNext import resnext50_2x64d
        net = resnext50_2x64d(num_classes=num_classes)
    elif args == 'resnext50_32x4d':
        from models.ResNext import resnext50_32x4d
        net = resnext50_32x4d(num_classes=num_classes)
    elif args == 'preactresnet18':
        from models.PreActRestNet import PreActResNet18
        net = PreActResNet18(num_classes=num_classes)
    elif args == 'preactresnet34':
        from models.PreActRestNet import PreActResNet34
        net = PreActResNet34(num_classes=num_classes)
    elif args == 'preactresnet50':
        from models.PreActRestNet import PreActResNet50
        net = PreActResNet50(num_classes=num_classes)
    elif args == 'preactresnet101':
        from models.PreActRestNet import PreActResNet101
        net = PreActResNet101(num_classes=num_classes)
    elif args == 'preactresnet152':
        from models.PreActRestNet import PreActResNet152
        net = PreActResNet152(num_classes=num_classes)
    elif args == 'wrn28-10':
        from models.WRN28_10 import wideResnet28_10
        net = wideResnet28_10(num_classes=num_classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, total_epochs=90, warmup_start_lr=0.1, warmup_end_lr=0.1,
                 milestones=[60, 120, 160], gamma=0.2, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.milestones = milestones
        self.gamma = gamma
        self.total_epochs = total_epochs
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:

            lr = self.warmup_start_lr + (self.warmup_end_lr - self.warmup_start_lr) * (
                        self.last_epoch / self.warmup_epochs)
        else:
            lr = self.warmup_end_lr
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    lr *= self.gamma
                else:
                    break
        return [lr for _ in self.base_lrs]

def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
