from Common.utils import evaluate, prRed, prepare_running
from Common.CommDatasets import load_datasets, partition_data
from torch.utils.data import DataLoader
from BaseFunc.CommModels import create_model_instance_SL
from MergeSFLClient import MergeSFLClient
import torch
from BaseFunc.Selector import uniform_select
def run():
    args, recorder, logger, client_manager, result_out = prepare_running()
    worker_num, active_num, device, epoch_lr = args.worker_num, args.active_num, args.device, args.lr

    train_dataset, test_dataset, class_num, resolution_per_sample = load_datasets('CIFAR100', args.data_path)
    client_datasets = partition_data(train_dataset, worker_num, args.data_pattern)

    # 一些全局掌握的变量
    client_profile_list = [client_manager.get_random_profile() for _ in range(worker_num)]
    train_loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    top_model_global, bottom_model_global = create_model_instance_SL(args.model_type, {'split_point': 10, 'class_num': class_num, 'pretrained': args.pretrained})
    criterion = torch.nn.CrossEntropyLoss()
    represented_clients = [MergeSFLClient() for _ in range(active_num)]

    # 初始化client的model
    for client in represented_clients:
        client.initialize_model(bottom_model_global)
    total_time = 0
    for cur_round in range(args.round):
        # 每轮递减learning rate
        if cur_round > 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))
        # 先进行client selection
        selected_client_ids = uniform_select(range(worker_num), active_num)
        clients_this_round = [represented_clients[i] for i in range(len(selected_client_ids))]
        run_samples_all_clients = []
        for i, client in enumerate(clients_this_round):
            # 对训练，首先对client进行设置，包括设置client_id, 复制全局模型, 设置optimizer, 设置loader
            client.set_client_id(selected_client_ids[i])
            client.load_model(bottom_model_global)
            client.set_optimizer(args.lr, args.momentum, args.weight_decay)
            client.set_train_dataloader(train_loaders[client.client_id])
            # 设置system properties
            client_profile: dict = client_profile_list[client.client_id]
            client.set_comm_bandwidth(client_profile['communication'] * 1000000)  # 原来是Mb/s
            client.set_compute_density(client_profile['computation'] * 1000000000)
        # 注意，这里的local_epoch和fedavg等不一致，这里每个local_epoch，client会执行一个batch
        for iter_idx in range(args.local_epoch):
            all_smashed_data = []
            all_targets = []
            all_detach_smashed_data = []
            for i, client in enumerate(clients_this_round):
                send_smash, smashed_data, targets = client.local_FP_training(device)
                all_detach_smashed_data.append(send_smash)
                all_smashed_data.append(smashed_data)
                all_targets.append(targets)

