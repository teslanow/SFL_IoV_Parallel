import os
from typing import List
from FedAvg.FedAvgClient import FedAvgClient
import numpy as np


def load_create_model_configs(model_type, class_num, pretrained):
    if model_type == 'Resnet34':
        return class_num, pretrained
    else:
        raise NotImplementedError

def Fedavg_runtime_per_round(all_clients : List[FedAvgClient], run_samples, macs_per_sample_training, parameters_of_model):
    """
    计算每个通信round的时间
    Args:
        parameters_of_model: 模型的参数量
        all_clients: List[Client]
        run_samples: 每个client在这个round训练的samples数目
        macs_per_sample_training: 每个sample的macs数
    Returns:

    """
    run_times = np.array([run_samples[i] * macs_per_sample_training / client.compute_density for i, client in enumerate(all_clients)] )
    comm_times = np.array([parameters_of_model * 2 * 4 / client.comm_bandwidth for client in all_clients] )
    return float(np.max(run_times + comm_times))