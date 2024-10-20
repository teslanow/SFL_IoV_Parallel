import copy
from torch import optim
from BaseFunc.ClientBase import ClientBase
class FedAvgClient(ClientBase):
    def __init__(self):
        super().__init__()


    def local_training(self, device, criterion=None):
        self.model.to(device)
        self.model.train()
        run_samples = 0
        if self.train_dataloader is not None:
            for inputs, labels in self.train_dataloader:
                # batch normalization的bug
                if inputs.size(0) <= 1:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # 统计数据
                run_samples += inputs.size(0)
        return run_samples

    def set_optimizer(self, lr, momentum, weight_decay):
        if momentum < 0:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum,
                                  nesterov=True, weight_decay=weight_decay)