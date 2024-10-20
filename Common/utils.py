import argparse
import json
import signal
from datetime import datetime
import logging
import os
import random

import numpy as np
import torch
import wandb
import yaml
from torch.utils.tensorboard import SummaryWriter
from typing import List
from Common.ClientProperties import ClientPropertyManager
import datasets
from Common.WandbWrapper import wandbInit, wandbFinishWrap

def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

def parse_args():
    # init parameters
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--dataset_type', type=str, default='CIFAR10')
    parser.add_argument('--model_type', type=str, default='Resnet34')
    parser.add_argument('--worker_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_pattern', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--decay_rate', type=float, default=0.993)
    parser.add_argument('--min_lr', type=float, default=0.005)
    parser.add_argument('--round', type=int, default=250, help='communication round')
    parser.add_argument('--local_epoch', type=int, default=1, help='local iteration')
    parser.add_argument('--momentum', type=float, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--use_cuda', action="store_false", default=True)
    parser.add_argument('--expname', type=str, default='MergeSFL')
    parser.add_argument('--active_num', type=int, default=1)
    parser.add_argument('--sys_conf_path', type=str, default='ExpConfig/System_conf.yml')
    # 指定一个同步轮训练中多少比例的clients能完成训练
    parser.add_argument('--finish_ratio', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrained', action='store_true', default=False, help='提供该参数，则表示开启预训练')

    return parser.parse_args()

def set_comm_config(common_config, args):
    common_config.model_type = args.model_type
    common_config.dataset_type = args.dataset_type
    common_config.batch_size = args.batch_size
    common_config.data_pattern = args.data_pattern
    common_config.lr = args.lr
    common_config.decay_rate = args.decay_rate
    common_config.min_lr = args.min_lr
    common_config.epoch = args.epoch
    common_config.momentum = args.momentum
    common_config.data_path = args.data_path
    common_config.weight_decay = args.weight_decay
    common_config.local_epoch = args.local_epoch

def get_client_logger(args, rank):
    RESULT_PATH = os.getcwd() + '/clients/' + args.expname + '/'

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)

    logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
    logger.setLevel(logging.INFO)

    filename = RESULT_PATH + os.path.basename(__file__).split('.')[0] + '_' + str(int(rank)) + '.log'
    fileHandler = logging.FileHandler(filename=filename)
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger

def set_recorder_and_logger(args):
    RESULT_PATH = os.getcwd() + '/server/'

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)

    cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    recorder: SummaryWriter = SummaryWriter(os.path.join('runs', args.expname, f"{args.finish_ratio}_{args.data_pattern}_{cur_time}"))

    logger = logging.getLogger(os.path.basename(__file__).split('.')[0])
    logger.setLevel(logging.INFO)

    # now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    filename = RESULT_PATH + args.expname + "_" + os.path.basename(__file__).split('.')[0] + '.log'
    fileHandler = logging.FileHandler(filename=filename)
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # logger.info(f"corresponding runtime file: {args.finish_ratio}_{args.data_pattern}_{cur_time}")

    return recorder, logger


class Statistics:
    def __init__(self):
        # 每个client在每一个同步round前的运行samples的数目
        self.run_samples_per_round_per_client = []
        # server在每一个同步round前的运行samples的数目
        self.run_samples_per_round_server = []
        # 每个client在每一个同步round前的通信的数据量
        self.comm_size_per_round_per_client = []

    def init_round(self, num_total_clients):
        self.run_samples_per_round_per_client.append([0 for _ in range(num_total_clients)])
        self.run_samples_per_round_server.append(0)
        self.comm_size_per_round_per_client.append([0 for _ in range(num_total_clients)])

    def update_client_sample(self, client_id, num_samples):
        self.run_samples_per_round_per_client[-1][client_id] = num_samples

    def update_server_sample(self, num_samples):
        self.run_samples_per_round_server[-1] = num_samples

    def update_client_comm_size(self, client_id, comm_size):
        self.comm_size_per_round_per_client[-1][client_id] = comm_size


class Sys_conf:
    def __init__(self):
        self.client_compute_density : List[float]  = None
        self.server_compute_density : List[float] = None
        self.client_communicate_bandwidth : List[float] = None

def load_system_config(path) -> Sys_conf:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    sys_conf = Sys_conf()
    for k, v in config.items():
        setattr(sys_conf, k, v)
    return sys_conf


def set_seed(self):
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num - 1))

    for i in range(train_class_num):
        partition_sizes[i][i % worker_num] = ratio

    return partition_sizes


def dirichlet_partition(dataset_type: str, alpha: float, worker_num: int, nclasses: int):
    partition_sizes = []
    filepath = './data_partition/%s-part_dir%.1f.npy' % (dataset_type, alpha)
    if os.path.exists(filepath):
        partition_sizes = np.load(filepath)
    else:
        for _ in range(nclasses):
            partition_sizes.append(np.random.dirichlet([alpha] * worker_num))
        partition_sizes = np.array(partition_sizes)
        # np.save(filepath, partition_sizes)

    return partition_sizes

def partition_data(dataset_type, data_pattern, data_path, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type, data_path)
    labels = None
    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num = 10

    elif dataset_type == "EMNIST":
        train_class_num = 62

    elif dataset_type == "CIFAR100" or dataset_type == "image100":
        train_class_num = 100

    elif dataset_type == "UCIHAR":
        train_class_num = 6

    elif dataset_type == "SPEECH":
        train_class_num = 35
        labels = sorted(list(set(datapoint[2] for datapoint in train_dataset)))

    if data_pattern == 0:
        partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
    elif data_pattern == 1:
        non_iid_ratio = 0.2
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 2:
        non_iid_ratio = 0.4
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 3:
        non_iid_ratio = 0.6
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 4:
        non_iid_ratio = 0.8
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)

    elif data_pattern == 5:  # dir-1.0
        print('Dirichlet partition 1.0')
        partition_sizes = dirichlet_partition(dataset_type, 1.0, worker_num, train_class_num)

    elif data_pattern == 6:  # dir-0.5
        print('Dirichlet partition 0.5')
        partition_sizes = dirichlet_partition(dataset_type, 0.5, worker_num, train_class_num)

    elif data_pattern == 7:  # dir-0.1
        print('Dirichlet partition 0.1')
        partition_sizes = dirichlet_partition(dataset_type, 0.1, worker_num, train_class_num)

    elif data_pattern == 8:  # dir-0.1
        print('Dirichlet partition 0.05')
        partition_sizes = dirichlet_partition(dataset_type, 0.01, worker_num, train_class_num)
    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes,
                                                         class_num=train_class_num, labels=labels)
    return train_dataset, test_dataset, train_data_partition, labels


def evaluate(model, test_loader, criterion, device):
    model.eval()
    model.to(device)
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def prepare_running():
    args = parse_args()
    # recorder, logger = set_recorder_and_logger(args)
    # path = os.getcwd()
    # print(path)
    # path = path + "//" + "result_recorder"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # path = path + "//" + now + "_record.txt"
    # result_out = open(path, 'w+')
    # result_out.write(f"\ncorresponding dir : {recorder.get_logdir()}\n")
    # json.dump(args.__dict__, result_out, indent=4)
    # print(args.__dict__, file=result_out)
    # result_out.write('\n')
    # result_out.write("epoch_idx, total_time, total_bandwith, total_resource, acc, test_loss")
    # result_out.write('\n')

    recorder, logger, result_out = None, None, None
    client_manager = ClientPropertyManager('ExpConfig/vehicle_device_capacity')
    # 设置signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # signal.signal(signal.SIGKILL, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)
    return args, recorder, logger, client_manager, result_out

def signal_handler(sig, frame):
    print(f'You pressed {sig}!')
    # 在这里执行清理操作
    wandbFinishWrap()
    exit()