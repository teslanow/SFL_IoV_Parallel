import datasets, models
import torch.optim as optim

class MS_Client:
    def __init__(self, common_config):
        self.train_loader = None
        self.optimizer = None
        self.epoch_lr = None
        self.common_config = common_config

    def initial_before_training(self, train_dataset,
                      labels, client_id, bsz, train_data_idxes, local_model, local_epoch):
        # lr
        self.epoch_lr = self.common_config.lr
        if local_epoch > 1:
            self.epoch_lr = max((self.common_config.decay_rate * self.epoch_lr, self.common_config.min_lr))
            self.common_config.lr = self.epoch_lr
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
            self.optimizer = optim.SGD(local_model.parameters(), lr=self.epoch_lr, weight_decay=self.common_config.weight_decay)
        else:
            self.optimizer = optim.SGD(local_model.parameters(), momentum=self.common_config.momentum, nesterov=True,
                                      lr=self.epoch_lr,
                                      weight_decay=self.common_config.weight_decay)


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