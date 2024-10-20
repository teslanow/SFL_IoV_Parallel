from torch import optim
from BaseFunc.ClientBase import ClientBase
class MergeSFLClient(ClientBase):
    def __init__(self):
        super().__init__()

    def set_optimizer(self, lr, momentum, weight_decay):
        if momentum < 0:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum,
                                  nesterov=True, weight_decay=weight_decay)

    def local_FP_training(self, device):
        self.model.train()
        self.model.to(device)
        inputs, targets = next(self.train_dataloader)
        # print(targets.shape)
        inputs = inputs.to(device)
        smashed_data = self.model(inputs)
        send_smash = smashed_data.clone().detach()
        return send_smash, smashed_data, targets
