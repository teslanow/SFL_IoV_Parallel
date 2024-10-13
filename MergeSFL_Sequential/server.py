import asyncio
import copy
from config import *
import datasets, models
import torch.optim as optim
from training_utils import *
from utils import *
from cient import *

args = parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda:7" if args.use_cuda and torch.cuda.is_available() else "cpu")

recorder, logger = set_recorder_and_logger(args)


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

    active_client_models = []
    for i in range(active_num):
        active_client_models.append(copy.deepcopy(global_client_model))

    active_clients = []
    for i in range(active_num):
        active_clients.append(MS_Client(common_config))

    client_init_para = torch.nn.utils.parameters_to_vector(global_client_model.parameters())
    client_model_size = client_init_para.nelement() * 4 / 1024 / 1024
    logger.info("para num: {}".format(client_init_para.nelement()))
    logger.info("Client Model Size: {} MB".format(client_model_size))

    global_init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    global_model_size = global_init_para.nelement() * 4 / 1024 / 1024
    logger.info("para num: {}".format(global_init_para.nelement()))
    logger.info("Global Server Model Size: {} MB".format(global_model_size))

    # Create model instance
    train_dataset, test_dataset, train_data_partition, labels = partition_data(args.dataset_type,
                                                                   args.data_pattern, args.data_path,
                                                                   worker_num)
    if labels:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False,
                                                  collate_fn=lambda x: datasets.collate_fn(x, labels))
    else:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    epoch_lr = args.lr

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
        selected_ids, bsz_list = control_seq(args.batch_size, active_num, worker_num)

        # 更新状态
        for i in range(active_num):
            cur_client_id = selected_ids[i]
            cur_bs = bsz_list[cur_client_id]
            train_data_idxes = train_data_partition.use(cur_client_id)
            active_clients[i].initial_before_training(train_dataset, labels, cur_client_id, cur_bs, train_data_idxes,
                                                      active_client_models[i], epoch_idx)

        prRed(selected_ids)

        local_steps = 42
        for iter_idx in range(local_steps):

            all_smashed_data = []
            all_targets = []
            all_detach_smashed_data = []

            # 所有clients的正向过程
            for i in range(active_num):
                client = active_clients[i]
                send_smash, smashed_data, targets = local_FP_training(active_client_models[i], device, client.train_loader)
                all_detach_smashed_data.append(send_smash)
                all_smashed_data.append(smashed_data)
                all_targets.append(targets)

            # split training
            train_loss, all_grads_in = merge_and_dispatch_seq(all_detach_smashed_data, all_targets, global_model, global_optim, selected_ids, bsz_list, device)

            # 所有clients的反向过程
            for i in range(active_num):
                client = active_clients[i]
                local_BP_training(all_grads_in[i], client.optimizer, all_smashed_data[i], device)

            print("\rstep: {} ".format(iter_idx + 1), end='', flush=True)
        print('\n')
        # 聚合
        result_para_state_dict = aggregate_model_dict(active_client_models, device)
        global_client_model.load_state_dict(result_para_state_dict)
        # evaluation
        test_loss, acc = test(global_model, global_client_model, test_loader, device)

        print("Epoch: {}, accuracy: {}, test_loss: {}".format(epoch_idx, acc, test_loss))
        result_out.write(
            '{} {:.2f} {:.2f} {:.2f} {:.4f} {:.4f}'.format(epoch_idx, 0, 0, 0, acc,
                                                           test_loss))
        result_out.write('\n')


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

def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num - 1))

    for i in range(train_class_num):
        partition_sizes[i][i % worker_num] = ratio

    return partition_sizes


def dirichlet_partition(dataset_type: str, alpha: float, worker_num: int, nclasses: int):
    partition_sizes = []
    filepath = './data_partition/%s-part_dir%.1f.npy' % (dataset_type, alpha)
    if os.path.exists(filepath):
        partition_sizes = np.load(filepath)
    else:
        for _ in range(nclasses):
            partition_sizes.append(np.random.dirichlet([alpha] * worker_num))
        partition_sizes = np.array(partition_sizes)
        # np.save(filepath, partition_sizes)

    return partition_sizes


def partition_data(dataset_type, data_pattern, data_path, worker_num=10):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type, data_path)
    labels = None
    if dataset_type == "CIFAR10" or dataset_type == "FashionMNIST":
        train_class_num = 10

    elif dataset_type == "EMNIST":
        train_class_num = 62

    elif dataset_type == "CIFAR100" or dataset_type == "image100":
        train_class_num = 100

    elif dataset_type == "UCIHAR":
        train_class_num = 6

    elif dataset_type == "SPEECH":
        train_class_num = 35
        labels = sorted(list(set(datapoint[2] for datapoint in train_dataset)))

    if data_pattern == 0:
        partition_sizes = np.ones((train_class_num, worker_num)) * (1.0 / worker_num)
    elif data_pattern == 1:
        non_iid_ratio = 0.2
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 2:
        non_iid_ratio = 0.4
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 3:
        non_iid_ratio = 0.6
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)
    elif data_pattern == 4:
        non_iid_ratio = 0.8
        partition_sizes = non_iid_partition(non_iid_ratio, train_class_num, worker_num)

    elif data_pattern == 5:  # dir-1.0
        print('Dirichlet partition 1.0')
        partition_sizes = dirichlet_partition(dataset_type, 1.0, worker_num, train_class_num)

    elif data_pattern == 6:  # dir-0.5
        print('Dirichlet partition 0.5')
        partition_sizes = dirichlet_partition(dataset_type, 0.5, worker_num, train_class_num)

    elif data_pattern == 7:  # dir-0.1
        print('Dirichlet partition 0.1')
        partition_sizes = dirichlet_partition(dataset_type, 0.1, worker_num, train_class_num)

    elif data_pattern == 8:  # dir-0.1
        print('Dirichlet partition 0.05')
        partition_sizes = dirichlet_partition(dataset_type, 0.01, worker_num, train_class_num)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes,
                                                         class_num=train_class_num, labels=labels)
    return train_dataset, test_dataset, train_data_partition, labels


if __name__ == "__main__":
    main()
