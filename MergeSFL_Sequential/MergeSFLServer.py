import copy
import torch
from torch import optim
from BaseFunc.ServerBase import ServerBase
import torch.nn.functional as F
class MergeSFLServer(ServerBase):
    def __init__(self, top_model_global, bottom_model_global):
        super().__init__()
        self.top_model_global = top_model_global
        self.bottom_model_global = bottom_model_global

    def merge_and_dispatch_seq(self, all_detach_smashed_data, all_targets, global_optim, bsz_list, device):
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
        global_optim.zero_grad()
        loss.backward()
        global_optim.step()

        # gradient dispatch
        sum_bsz = sum(bsz_list)
        bsz_s = 0
        all_grad_ins = []
        for i, batch_size in enumerate(bsz_list):
            grad_in = (m_data.grad[bsz_s: bsz_s + bsz_list[i]] * sum_bsz / bsz_list[i])
            bsz_s += bsz_list[i]
            all_grad_ins.append(grad_in.clone().detach())

        return loss.item(), all_grad_ins

