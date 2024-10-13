import torch.nn

import datasets, models
import torch.optim as optim

class MS_Client:
    def __init__(self, common_config):
        self.bsz = None
        self.train_loader = None
        self.optimizer = None
        self.epoch_lr = None
        self.common_config = common_config

        # system properties
        self.compute_density = None
        self.communicate_bandwidth = None

        # 存在状态
        self.status = 0 # 0表示不允许，1表示运行
        self.client_id = None

        # 执行steps
        self.local_steps = 0

        # model
        self.model : torch.nn.Module = None

    def initial_before_training(self, train_dataset,
                      labels, client_id, bsz, train_data_idxes, local_epoch):
        self.client_id = client_id
        # lr
        self.epoch_lr = self.common_config.lr
        self.bsz = bsz
        if local_epoch > 1:
            self.epoch_lr = max((self.common_config.decay_rate * self.epoch_lr, self.common_config.min_lr))
            self.common_config.lr = self.epoch_lr
        del self.train_loader
        if labels:
            self.train_loader = datasets.create_dataloaders(train_dataset, batch_size=int(bsz),
                                                       selected_idxs=train_data_idxes,
                                                       pin_memory=False, drop_last=True,
                                                       collate_fn=lambda x: datasets.collate_fn(x, labels))
        else:
            self.train_loader = datasets.create_dataloaders(train_dataset, batch_size=int(bsz),
                                                       selected_idxs=train_data_idxes,
                                                       pin_memory=False, drop_last=True)

        # print("vid({}) epoch-{} lr: {} bsz: {} ".format(client_id, local_epoch, self.epoch_lr, bsz))


        if self.common_config.momentum < 0:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.epoch_lr, weight_decay=self.common_config.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), momentum=self.common_config.momentum, nesterov=True,
                                      lr=self.epoch_lr,
                                      weight_decay=self.common_config.weight_decay)

    def update_compute_properties(self, compute_density):
        """
        Args:
            compute_density: unit in flop/s
        """
        self.compute_density = compute_density

    def update_communicate_bandwidth(self, communicate_bandwidth):
        """
        Args:
            communicate_bandwidth: unit in B/s
        """
        self.communicate_bandwidth = communicate_bandwidth

def local_FP_training(local_model, device, train_loader):
    local_model.train()
    local_model.to(device)
    inputs, targets = next(train_loader)
    inputs = inputs.to(device)
    smashed_data = local_model(inputs)
    send_smash = smashed_data.clone().detach()

    return send_smash, smashed_data, targets


def local_BP_training(grad_in, optimizer, smashed_data, device):
    grad_in = grad_in.to(device)
    smashed_data = smashed_data.to(device)
    optimizer.zero_grad()
    smashed_data.backward(grad_in.to(device))
    optimizer.step()