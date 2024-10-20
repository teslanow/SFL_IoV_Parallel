import copy

import torch
import torch.nn as nn

class ClientBase:
    def __init__(self):
        super().__init__()
        # 单位B/s
        self.comm_bandwidth = None
        # 单位flop/s
        self.compute_density = None
        self.client_id = None
        self.optimizer = None
        self.train_dataloader = None
        self.model: nn.Module = None

    def local_training(self, device, criterion):
        """

        Returns: local_training所用的时间

        """
        raise NotImplementedError

    def set_compute_density(self, compute_density):
        """
        暂时先用计算速度Flops代替，后续应该使用更细的粒度
        Args:
            compute_density: unit in Flops
        """
        self.compute_density = compute_density

    def set_comm_bandwidth(self, comm_bandwidth):
        """
        Args:
            comm_bandwidth: unit in byte/s
        """
        self.comm_bandwidth = comm_bandwidth

    def time_to_send_tensor(self, tensor: torch.Tensor, byte_per_element, receiver):
        """
        receiver暂时不用
        Args:
            tensor:
            byte_per_element: tensor每个element所占的字节数
            receiver:
        Returns: time, unit in second
        """
        return tensor.numel() * byte_per_element / self.comm_bandwidth

    def set_train_dataloader(self, train_dataloader):
        self.train_dataloader = train_dataloader

    def set_client_id(self, client_id):
        self.client_id = client_id

    def initialize_model(self, model: nn.Module):
        """
        从model拷贝一份给自己，这是新建操作
        """
        self.model = copy.deepcopy(model)

    def load_model(self, model: nn.Module):
        """
        将model的参数复制到自己的model里
        """
        self.model.load_state_dict(model.state_dict())