import argparse
from datetime import datetime
import logging
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from typing import List
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

def parse_args():
    # init parameters
    parser = argparse.ArgumentParser(description='Distributed Client')
    parser.add_argument('--dataset_type', type=str, default='CIFAR10')
    parser.add_argument('--model_type', type=str, default='AlexNet')
    parser.add_argument('--worker_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_pattern', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--decay_rate', type=float, default=0.993)
    parser.add_argument('--min_lr', type=float, default=0.005)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--momentum', type=float, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--use_cuda', action="store_false", default=True)
    parser.add_argument('--expname', type=str, default='MergeSFL')
    parser.add_argument('--local_epoch', type=int, default=42)
    parser.add_argument('--active_num', type=int, default=1)
    parser.add_argument('--sys_conf_path', type=str, default='ExpConfig/System_conf.yml')
    # 指定一个同步轮训练中多少比例的clients能完成训练
    parser.add_argument('--finish_ratio', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
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
