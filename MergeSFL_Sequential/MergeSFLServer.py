import copy
from typing import List
import numpy as np
import torch
from torch import optim
from BaseFunc.ServerBase import ServerBase
import torch.nn.functional as F
class MergeSFLServer(ServerBase):
    def __init__(self, top_model_global, bottom_model_global):
        super().__init__()
        self.top_model_global = top_model_global
        self.bottom_model_global = bottom_model_global

    def set_optimizer(self, lr, momentum, weight_decay):
        if momentum < 0:
            self.optimizer = optim.SGD(self.top_model_global.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(self.top_model_global.parameters(), lr=lr, momentum=momentum,
                                  nesterov=True, weight_decay=weight_decay)

    def merge_and_dispatch_seq(self, all_detach_smashed_data, all_targets, bsz_list, device):
        """
        先merge所有的batch，然后进行global model的forward和backward
        Args:
            all_detach_smashed_data: 所有当前iteration上传的smashed data集合
            all_targets:：所有对应的targets
            global_optim:
            bsz_list:每个client的sample size
            device:

        Returns:

        """
        m_data = all_detach_smashed_data
        m_target = all_targets
        m_data = torch.cat(m_data, dim=0)
        m_target = torch.cat(m_target, dim=0)

        m_data = m_data.to(device)
        m_target = m_target.to(device)
        global_model = self.top_model_global.to(device)

        m_data.requires_grad_()

        # server side fp
        outputs = global_model(m_data)
        loss = F.cross_entropy(outputs, m_target.long())

        # server side bp
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # gradient dispatch
        sum_bsz = sum(bsz_list)
        bsz_s = 0
        all_grad_ins = []
        for i, batch_size in enumerate(bsz_list):
            grad_in = (m_data.grad[bsz_s: bsz_s + bsz_list[i]] * sum_bsz / bsz_list[i])
            bsz_s += bsz_list[i]
            all_grad_ins.append(grad_in.clone().detach())

        return loss.item(), all_grad_ins

    def aggregate_model_dict(self, active_models: List[torch.nn.Module], device):
        """
        聚合，目前少了batch size作为weight
        Args:
            active_models:
            device:

        Returns:

        """
        # return state_dict
        with torch.no_grad():
            para_delta = copy.deepcopy(active_models[0].state_dict())
            for para in para_delta.keys():
                para_delta[para] = para_delta[para].to(device)
                for i in range(1, len(active_models)):
                    para_delta[para] += active_models[i].state_dict()[para].to(device)
                para_delta[para] = torch.div(para_delta[para], len(active_models))
            self.bottom_model_global.load_state_dict(para_delta)

    def evaluate(self, data_loader, device):
        self.bottom_model_global.to(device)
        self.top_model_global.to(device)
        self.bottom_model_global.eval()
        self.top_model_global.eval()

        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)

                output1 = self.bottom_model_global(data)
                output = self.top_model_global(output1)

                # sum up batch loss
                loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
                test_loss += loss_func(output, target.long()).item()

                pred = output.argmax(1, keepdim=True)
                batch_correct = pred.eq(target.view_as(pred)).sum().item()

                correct += batch_correct

        test_loss /= len(data_loader.dataset)
        test_accuracy = np.float(1.0 * correct / len(data_loader.dataset)) * 100
        return test_loss, test_accuracy
