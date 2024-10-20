import numpy as np
import torch
from typing import List
from torch.distributions import Dirichlet
from torch.utils.data import random_split
from torchvision import datasets, transforms


def load_default_transform(dataset_type, train=False):
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        if train:
            dataset_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize
            ])
        else:
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

    elif dataset_type == 'CIFAR100':
        # reference:https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        if train:
            dataset_transform = transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
        else:
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

    elif dataset_type == 'FashionMNIST':
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset_type == 'MNIST':
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset_type == 'SVHN':
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif dataset_type == 'EMNIST':
        dataset_transform = transforms.Compose([
            transforms.Resize(28),
            # transforms.CenterCrop(227),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    elif dataset_type == 'tinyImageNet':
        dataset_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

    elif dataset_type == 'image100':
        dataset_transform = transforms.Compose([transforms.Resize((144, 144)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                ])
    else:
        dataset_transform = None

    return dataset_transform

def load_datasets(dataset_type, data_path):
    train_transform = load_default_transform(dataset_type, train=True)
    test_transform = load_default_transform(dataset_type, train=False)

    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transform)
        class_num = 10
        resolution_per_sample = (3, 32, 32)
    elif dataset_type == 'CIFAR100':
        train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=test_transform)
        class_num = 100
        resolution_per_sample = (3, 32, 32)
    elif dataset_type == 'tinyImageNet':
        train_dataset = datasets.ImageFolder(data_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(data_path, transform=train_transform)
        class_num = 0
        resolution_per_sample = (3, 32, 32)
    else:
        raise NotImplementedError
    return train_dataset, test_dataset, class_num, resolution_per_sample

def get_resolution_of_dataset(dataset_type):
    """
    返回各种dataset的一个batch的sample尺寸，用于计算model density
    Args:
        dataset_type: 见上面
    Returns: Tuple[int,int, ...]

    """

    if dataset_type == 'CIFAR10':
        return 3, 32, 32
    elif dataset_type == 'CIFAR100':
        return 3, 32, 32
    elif dataset_type == 'tinyImageNet':
        return 3, 32, 32
    else:
        raise NotImplementedError

def dirichlet_partition(alpha: float, worker_num: int) -> List[float]:
    # 使用 Dirichlet 分布划分数据集
    # 翻译
    dirichlet_dist = Dirichlet(torch.tensor([alpha] * worker_num))
    return dirichlet_dist.sample().tolist()

def partition_data(train_dataset:datasets, worker_num:int, data_pattern:int):
    """
    划分训练数据
    Args:
        train_dataset: datasets类型
        worker_num: 划分的份数
        data_pattern: 根据pattern的数值确定划分的方式
    Returns: 划分的datasets, 划分的list
    """
    if data_pattern == 0:
        partition_sizes = (np.ones(worker_num) / worker_num).tolist()

    elif data_pattern == 5:  # dir-1.0
        print('Dirichlet partition 1.0')
        partition_sizes = dirichlet_partition(1.0, worker_num)

    elif data_pattern == 6:  # dir-0.5
        print('Dirichlet partition 0.5')
        partition_sizes = dirichlet_partition(0.5, worker_num)

    elif data_pattern == 7:  # dir-0.1
        print('Dirichlet partition 0.1')
        partition_sizes = dirichlet_partition(0.1, worker_num)

    elif data_pattern == 8:  # dir-0.1
        print('Dirichlet partition 0.05')
        partition_sizes = dirichlet_partition(0.01, worker_num)
    else:
        raise NotImplementedError

    size_dataset = len(train_dataset)
    split_size = []
    left_size = size_dataset
    for ratio in partition_sizes:
        cur_len = min(int(size_dataset * ratio), left_size)
        split_size.append(cur_len)
        left_size -= cur_len
    if left_size != 0:
        split_size[-1] += left_size
    client_datasets = random_split(train_dataset, split_size)
    return client_datasets
