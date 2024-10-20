import copy

import datasets
from models import *
from config import *
from training_utils import *
from Common.utils import *
from .cient import *
set_seed(1234)
args = parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device(args.device)

sys_conf = load_system_config(args.sys_conf_path)

recorder, logger = set_recorder_and_logger(args)

server_compute_density = random.choice(sys_conf.server_compute_density)

def main():
    active_num = args.active_num
    worker_num = args.worker_num

    common_config = CommonConfig()
    set_comm_config(common_config, args)

    path = os.getcwd()
    print(path)
    path = path + "//" + "result_recorder"
    if not os.path.exists(path):
        os.makedirs(path)

    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    path = path + "//" + now + "_record.txt"
    result_out = open(path, 'w+')
    print(args.__dict__, file=result_out)
    result_out.write('\n')
    result_out.write("epoch_idx, total_time, total_bandwith, total_resource, acc, test_loss")
    result_out.write('\n')

    global_client_model, global_model = models.create_model_instance_SL(args.dataset_type, args.model_type)


    active_clients = []
    for i in range(active_num):
        active_clients.append(MS_Client(common_config))
        active_clients[-1].model = copy.deepcopy(global_client_model)

    client_init_para = torch.nn.utils.parameters_to_vector(global_client_model.parameters())
    client_model_size = client_init_para.nelement() * 4
    logger.info("para num: {}".format(client_init_para.nelement()))
    logger.info("Client Model Size: {} MB".format(client_model_size / 1024 / 1024))

    global_init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    global_model_size = global_init_para.nelement() * 4
    logger.info("para num: {}".format(global_init_para.nelement()))
    logger.info("Global Server Model Size: {} MB".format(global_model_size / 1024 / 1024))

    # Create model instance
    train_dataset, test_dataset, train_data_partition, labels = partition_data(args.dataset_type,
                                                                   args.data_pattern, args.data_path,
                                                                   worker_num)
    # 去除空的partition

    test_loader = datasets.create_dataloaders_without_helpler(test_dataset, batch_size=128, shuffle=False)

    epoch_lr = args.lr

    cur_time = 0

    # 异常检查，因为partition有可能为空
    worker_list = [index for index in range(worker_num) if len(train_data_partition.use(index)) > 0]

    for epoch_idx in range(1, 1 + args.epoch):
        # learning rate
        if epoch_idx > 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))

        if args.momentum < 0:
            global_optim = optim.SGD(global_model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
        else:
            global_optim = optim.SGD(global_model.parameters(), lr=epoch_lr, momentum=args.momentum,
                                     nesterov=True, weight_decay=args.weight_decay)

        # selection strategy and batch size configuration for all virtual workers
        selected_ids, bsz_list = control_seq_2(args.batch_size, active_num, worker_list, worker_num)

        id2client = dict()

        # 更新状态
        for i in range(len(selected_ids)):
            cur_client_id = selected_ids[i]
            cur_bs = bsz_list[cur_client_id]
            train_data_idxes = train_data_partition.use(cur_client_id)
            client = active_clients[i]
            client.model.load_state_dict(state_dict=global_client_model.state_dict())
            client.initial_before_training(train_dataset, labels, cur_client_id, cur_bs, train_data_idxes, epoch_idx)

            # 设置client的system config
            client.update_compute_properties(random.choice(sys_conf.client_compute_density))
            client.update_communicate_bandwidth(random.choice(sys_conf.client_communicate_bandwidth))
            id2client[cur_client_id] = client

        comm_time = 0

        # 计算模型下载时间
        parallel_param_comm = np.array([client_model_size for _ in range(len(active_clients))])
        cur_time += np.max(parallel_param_comm / np.array(
            [active_clients[i].communicate_bandwidth for i in range(len(active_clients))]))
        comm_time += np.max(parallel_param_comm / np.array(
            [active_clients[i].communicate_bandwidth for i in range(len(active_clients))]))
        prRed(selected_ids)
        local_steps = 42

        num_cannot_finish = len(selected_ids) - min(len(selected_ids), math.ceil(len(selected_ids) * args.finish_ratio))
        ids_cannot_finish = random.sample(selected_ids, num_cannot_finish)
        steps_of_failed_ids = np.random.choice(range(local_steps - 1), size=num_cannot_finish)

        for client in active_clients:
            client.local_steps = local_steps
        for i, t in enumerate(ids_cannot_finish):
            id2client[t].local_steps = steps_of_failed_ids[i]

        t_active_clients = active_clients



        for iter_idx in range(local_steps):

            l = len(t_active_clients)
            t_active_clients = [client for client in t_active_clients if client.local_steps >= iter_idx]
            if len(t_active_clients) != l:
                prGreen([c.client_id for c in t_active_clients])

            all_smashed_data = []
            all_targets = []
            all_detach_smashed_data = []

            comm_size_client2server = []

            # 所有clients的正向过程
            for i in range(len(t_active_clients)):
                client = t_active_clients[i]
                send_smash, smashed_data, targets = local_FP_training(client.model, device, client.train_loader)
                comm_size_client2server.append(send_smash.numel() * 4 + targets.numel() * 2)
                all_detach_smashed_data.append(send_smash)
                all_smashed_data.append(smashed_data)
                all_targets.append(targets)

            # split training
            train_loss, all_grads_in = merge_and_dispatch_seq(all_detach_smashed_data, all_targets, global_model, global_optim, selected_ids, bsz_list, device)

            comm_size_server2client = []
            # 所有clients的反向过程
            for i in range(len(t_active_clients)):
                client = t_active_clients[i]
                comm_size_server2client.append(all_grads_in[i].numel() * 4)
                local_BP_training(all_grads_in[i], client.optimizer, all_smashed_data[i], device)

            # 以下是统计数据的模拟计算
            parallel_FP_compute_times = np.array(simulate_client_FP_compute(t_active_clients))
            parallel_comm_client2server = np.array(comm_size_client2server) / np.array([t_active_clients[i].communicate_bandwidth for i in range(len(t_active_clients))])
            parallel_comm_server2client = np.array(comm_size_server2client) / np.array([t_active_clients[i].communicate_bandwidth for i in range(len(t_active_clients))])
            parallel_BP_compute_times = np.array(simulate_client_BP_compute(t_active_clients))

            forward_time = np.max(parallel_FP_compute_times + parallel_comm_client2server) + simulate_server_FP_compute(t_active_clients)
            backward_time = simulate_server_BP_compute(t_active_clients) + np.max(parallel_comm_server2client + parallel_BP_compute_times)

            comm_time += ((np.average(parallel_comm_client2server)) + (np.average(parallel_comm_server2client)))

            cur_time += (forward_time + backward_time)

            print("\rstep: {} ".format(iter_idx + 1), end='', flush=True)
        print('\n')
        # 聚合
        result_para_state_dict = aggregate_model_dict([client.model for client in t_active_clients], device)
        global_client_model.load_state_dict(result_para_state_dict)

        # 统计参数上传时间
        parallel_param_comm = np.array([client_model_size for _ in range(len(t_active_clients))])

        comm_time += np.max(parallel_param_comm / np.array([t_active_clients[i].communicate_bandwidth for i in range(len(t_active_clients))]))
        cur_time += np.max(parallel_param_comm / np.array([t_active_clients[i].communicate_bandwidth for i in range(len(t_active_clients))]))

        print(comm_time)
        print(cur_time)
        # evaluation
        test_loss, acc = test(global_model, global_client_model, test_loader, device)

        recorder.add_scalar('Train/time', cur_time, epoch_idx)
        recorder.add_scalar('Test/acc', acc, epoch_idx)
        recorder.add_scalar('Test/loss', test_loss, epoch_idx)
        recorder.add_scalar('Test/acc-time', acc, cur_time)
        recorder.add_scalar('Train/comm_cost', 0, epoch_idx)
        prRed("Epoch: {}, curtime: {}, accuracy: {}, test_loss: {}".format(epoch_idx, cur_time ,acc, test_loss))
        result_out.write(
            '{} {:.2f} {:.2f} {:.2f} {:.4f} {:.4f}'.format(epoch_idx, cur_time, 0, 0, acc,
                                                           test_loss))
        result_out.write('\n')


def simulate_client_FP_compute(active_clients: List[MS_Client]):
    run_times = [client.bsz * FP_density_AlexNet_DF1_CIFAR10 / client.compute_density for client in active_clients]
    return run_times

def simulate_client_BP_compute(active_clients: List[MS_Client]):
    run_times = [client.bsz * BP_density_AlexNet_DF1_CIFAR10 / client.compute_density for client in active_clients]
    return run_times

def simulate_server_FP_compute(active_clients: List[MS_Client]):
    bszs = [client.bsz for client in active_clients]
    total_bs = sum(bszs)
    return total_bs * FP_density_AlexNet_DF2_CIFAR10 / server_compute_density

def simulate_server_BP_compute(active_clients: List[MS_Client]):
    bszs = [client.bsz for client in active_clients]
    total_bs = sum(bszs)
    return total_bs * BP_density_AlexNet_DF2_CIFAR10 / server_compute_density

def aggregate_model_dict(active_models:List[torch.nn.Module], device):
    # return state_dict
    with torch.no_grad():
        para_delta = copy.deepcopy(active_models[0].state_dict())
        for para in para_delta.keys():
            para_delta[para] = para_delta[para].to(device)
            for i in range(1, len(active_models)):
                para_delta[para] += active_models[i].state_dict()[para].to(device)
            para_delta[para] = torch.div(para_delta[para], len(active_models))
    return para_delta



if __name__ == "__main__":
    main()
