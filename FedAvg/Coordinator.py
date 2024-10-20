from FedAvg.FedAvgUtil import load_create_model_configs, Fedavg_runtime_per_round
from Common.CommDatasets import load_datasets, partition_data
from torch.utils.data import DataLoader
from BaseFunc.CommModels import create_model_full, model_density_per_layer
from FedAvg.FedAvgClient import FedAvgClient
from BaseFunc.Selector import uniform_select
from FedAvg.Aggregation import aggregate_model_dict
import torch
from Common.utils import evaluate, prRed, prepare_running
from Common.WandbWrapper import wandbFinishWrap, wandbLogWrap, wandbInit
def run():

    args, recorder, logger, client_manager, result_out = prepare_running()
    wandbInit(args)
    worker_num, active_num, device, epoch_lr = args.worker_num, args.active_num, args.device, args.lr

    train_dataset, test_dataset, class_num, resolution_per_sample = load_datasets(args.dataset_type, args.data_path)
    client_datasets = partition_data(train_dataset, worker_num, args.data_pattern)

    # 一些全局掌握的变量
    client_profile_list = [client_manager.get_random_profile() for _ in range(worker_num)]
    train_loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=True) if len(ds) > 0 else None  for ds in client_datasets ]
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    config = load_create_model_configs(args.model_type, class_num, args.pretrained)
    model_global = create_model_full(args.model_type, config)
    # 用乘以三简略估计
    macs_per_sample_training, parameters_of_model = model_density_per_layer(model_global, resolution_per_sample)
    macs_per_sample_training = macs_per_sample_training * 3
    criterion = torch.nn.CrossEntropyLoss()
    represented_clients = [FedAvgClient() for _ in range(active_num)]

    # 初始化client的model
    for client in represented_clients:
        client.initialize_model(model_global)
    total_time = 0
    # 开始协调训练
    for cur_round in range(args.round):
        if cur_round >= 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))
        # 先进行client selection
        selected_client_ids = uniform_select(range(worker_num), active_num)
        clients_this_round = [represented_clients[i] for i in range(len(selected_client_ids))]
        run_samples_all_clients = []
        for i, client in enumerate(clients_this_round):
            # 对训练，首先对client进行设置，包括设置client_id, 复制全局模型, 设置optimizer, 设置loader
            client.set_client_id(selected_client_ids[i])
            client.load_model(model_global)
            client.set_optimizer(epoch_lr, args.momentum, args.weight_decay)
            client.set_train_dataloader(train_loaders[client.client_id])
            # 设置system properties
            client_profile: dict = client_profile_list[client.client_id]
            client.set_comm_bandwidth(client_profile['communication'] * 1000000)    # 原来是Mb/s
            client.set_compute_density(client_profile['computation'] * 1000000000)    # 原来是GFlop/s

            # 开始local_training
            run_samples = 0
            for _ in range(args.local_epoch):
                run_samples += client.local_training(device, criterion)
            run_samples_all_clients.append(run_samples)
        # 开始聚合
        result_state_dict = aggregate_model_dict([c.model for c in clients_this_round], device, run_samples_all_clients)
        model_global.load_state_dict(result_state_dict)

        # 开始测试
        test_loss, test_acc = evaluate(model_global, test_loader, criterion, device)
        # 计算这个round的时间
        run_time_this_round = Fedavg_runtime_per_round(clients_this_round, run_samples_all_clients, macs_per_sample_training, parameters_of_model)
        total_time += run_time_this_round
        prRed('Round = {}, runtime = {}, Test Loss = {}, Test Acc = {}%'.format(cur_round, total_time, test_loss, test_acc))
        wandbLogWrap({
            "Round" : cur_round,
            "total_time" : total_time,
            "test_loss" : test_loss,
            "test_acc" : test_acc,
            "epoch_lr" : epoch_lr
        })
    wandbFinishWrap()
        # recorder.add_scalar('Train/time', total_time, cur_round)
        # recorder.add_scalar('Test/acc', test_acc, cur_round)
        # recorder.add_scalar('Test/loss', test_loss, cur_round)
        # recorder.add_scalar('Test/acc-time', test_acc, total_time)
        # result_out.write('{} {:.2f} {:.2f} {:.2f} {:.4f} {:.4f}'.format(cur_round, total_time, 0, 0, test_acc,test_loss))
        # result_out.write('\n')



