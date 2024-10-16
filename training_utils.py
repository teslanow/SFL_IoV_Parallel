import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def merge_and_dispatch_seq(all_detach_smashed_data, all_targets, global_model, global_optim, selected_ids, bsz_list,
                       device=torch.device('cpu')):
    """
    merge collected batches for server-side training

    Parameters:
        global_model: server-side model
        global_optim: server-side optimizer
        bsz_list: list of batchsizes for all workers

    Return:
        average training loss
        comm cost
    """

    comm_cost = 0
    # merge smashed data
    m_data = all_detach_smashed_data
    m_target = all_targets

    m_data = torch.cat(m_data, dim=0)
    m_target = torch.cat(m_target, dim=0)

    m_data = m_data.to(device)
    m_target = m_target.to(device)
    global_model = global_model.to(device)

    m_data.requires_grad_()

    # server side fp
    outputs = global_model(m_data)
    loss = F.cross_entropy(outputs, m_target.long())

    # server side bp
    global_optim.zero_grad()
    loss.backward()
    global_optim.step()

    # gradient dispatch
    sum_bsz = sum([bsz_list[i] for i in selected_ids])
    bsz_s = 0
    all_grad_ins = []
    for worker_idx in selected_ids:
        grad_in = (m_data.grad[bsz_s: bsz_s + bsz_list[worker_idx]] * sum_bsz / bsz_list[worker_idx])
        bsz_s += bsz_list[worker_idx]
        all_grad_ins.append(grad_in.clone().detach())

    return loss.item(), all_grad_ins


def merge_and_dispatch(args, global_model, global_optim, selected_ids, bsz_list, worker_list, device=torch.device('cpu')):
    """
    merge collected batches for server-side training

    Parameters:
        global_model: server-side model  
        global_optim: server-side optimizer
        bsz_list: list of batchsizes for all workers
    
    Return:
        average training loss
        comm cost
    """
    comm_cost = 0
    # merge smashed data
    m_data = []
    m_target = []
    for worker_id, worker in zip(selected_ids, worker_list):
        data_i, target_i = worker.config.neighbor_paras[0].to(device), worker.config.neighbor_paras[1].to(device)
        m_data.append(data_i)
        m_target.append(target_i)
        comm_cost += data_i.nelement() * 4 + target_i.nelement() * 2

    m_data = torch.cat(m_data, dim=0)
    m_target = torch.cat(m_target, dim=0)
    
    m_data.requires_grad_()

    # server side fp
    outputs = global_model(m_data)
    loss = F.cross_entropy(outputs, m_target.long())

    # server side bp
    global_optim.zero_grad()
    loss.backward()
    global_optim.step()
    
    # gradient dispatch
    sum_bsz = sum([bsz_list[i] for i in selected_ids])
    bsz_s = 0
    for worker_idx, worker in zip(selected_ids, worker_list):
        worker.config.grad_in = (m_data.grad[bsz_s: bsz_s + bsz_list[worker_idx]] * sum_bsz / bsz_list[worker_idx])
        bsz_s += bsz_list[worker_idx]
        comm_cost += worker.config.grad_in.nelement() * 4
          
    return loss.item(), comm_cost


def control(args, active_num, worker_list):

    selected_ids = np.random.choice(range(len(worker_list)), size=active_num, replace=False)
    # bsz_list = np.random.randint(32, 64 + 1, len(worker_list))
    bsz_list = np.ones(len(worker_list), dtype=int) * args.batch_size
    
    return selected_ids, bsz_list


def control_seq(batch_size, active_num, worker_num):
    selected_ids = np.random.choice(range(worker_num), size=active_num, replace=False)
    # bsz_list = np.random.randint(32, 64 + 1, len(worker_list))
    bsz_list = np.ones(worker_num, dtype=int) * batch_size

    return selected_ids.tolist(), bsz_list.tolist()

def control_seq_2(batch_size, active_num, worker_list, worker_num):
    selected_ids = random.sample(worker_list, min(len(worker_list), active_num))
    # bsz_list = np.random.randint(32, 64 + 1, len(worker_list))
    bsz_list = np.ones(worker_num, dtype=int) * batch_size

    return selected_ids, bsz_list.tolist()


def test2(model, data_loader, device):
    model.to(device)
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            # sum up batch loss
            loss_func = nn.CrossEntropyLoss(reduction='sum') 
            test_loss += loss_func(output, target.long()).item()

            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct
            
    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))
    return test_loss, test_accuracy


def test(model_p, model_fe, data_loader, device=torch.device("cpu")):

    model_fe.to(device)
    model_p.to(device)

    model_p.eval()
    model_fe.eval()
    data_loader = data_loader.loader
    test_loss = 0.0
    test_accuracy = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            output1 = model_fe(data)
            output = model_p(output1)

            # sum up batch loss
            loss_func = nn.CrossEntropyLoss(reduction='sum')
            test_loss += loss_func(output, target.long()).item()

            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()

            correct += batch_correct

    test_loss /= len(data_loader.dataset)
    test_accuracy = np.float(1.0 * correct / len(data_loader.dataset))
    return test_loss, test_accuracy