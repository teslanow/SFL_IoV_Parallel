import asyncio
import copy
from config import *
import models
import torch.optim as optim
from mpi4py import MPI
from training_utils import *
from Common.utils import *

args = parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda:7" if args.use_cuda and torch.cuda.is_available() else "cpu")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
csize = comm.Get_size()

recorder, logger = set_recorder_and_logger(args)


def server_training(args, global_model, global_optim, selected_ids, bsz_list, worker_list,
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
    total_loss = 0.0

    for worker_id, worker in zip(selected_ids, worker_list):
        data, target = worker.config.neighbor_paras[0].to(device), worker.config.neighbor_paras[1].to(device)
        global_model.to(device)
        data.requires_grad_()
        outputs = global_model(data)
        loss = F.cross_entropy(outputs, target.long())
        global_optim.zero_grad()
        loss.backward()
        global_optim.step()
        worker.config.grad_in = data.grad

        comm_cost += (data.nelement() * 4 + target.nelement() * 2 + worker.config.grad_in.nelement() * 4)
        total_loss += loss.item()

    return total_loss / len(selected_ids), comm_cost

def main():
    logger.info("csize:{}".format(int(csize)))
    logger.info("server start (rank):{}".format(int(rank)))
    # init config
    common_config = CommonConfig()
    set_comm_config(common_config, args)

    active_num = int(csize) - 1
    worker_num = args.worker_num

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

    # global_model = models.create_model_instance(common_config.dataset_type, common_config.model_type)
    client_model, global_model = models.create_model_instance_SL(common_config.dataset_type, common_config.model_type)
    client_init_para = torch.nn.utils.parameters_to_vector(client_model.parameters())

    global_model.to(device)
    client_model.to(device)

    common_config.para_nums = client_init_para.nelement()
    client_model_size = client_init_para.nelement() * 4 / 1024 / 1024
    logger.info("para num: {}".format(common_config.para_nums))
    logger.info("Client Model Size: {} MB".format(client_model_size))

    global_init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    global_model_size = global_init_para.nelement() * 4 / 1024 / 1024
    logger.info("para num: {}".format(global_init_para.nelement()))
    logger.info("Global Model Size: {} MB".format(global_model_size))

    # Create model instance
    _, test_dataset, train_data_partition, labels = partition_data(common_config.dataset_type,
                                                                   common_config.data_pattern, common_config.data_path,
                                                                   worker_num)
    common_config.labels = labels

    # create active workers
    worker_list: List[Worker] = list()
    for worker_idx in range(active_num):
        worker_list.append(
            Worker(config=ClientConfig(common_config=common_config), rank=worker_idx + 1)
        )

    # set dynamic context for virtual workers
    virtual_worker_list: List[Worker] = list()
    for worker_idx in range(worker_num):
        virtual_worker_list.append(
            Worker(config=ClientConfig(common_config=common_config), rank=worker_idx + 1)
        )
        train_data_idxes = train_data_partition.use(worker_idx)
        virtual_worker_list[-1].config.train_data_idxes = train_data_idxes

    # connect socket and send init config
    communication_parallel(worker_list, 1, comm, action="init", selected_ids=np.arange(len(worker_list)))

    # recorder: SummaryWriter = SummaryWriter()
    global_model.to(device)

    if labels:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False,
                                                  collate_fn=lambda x: datasets.collate_fn(x, labels))

    else:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    global_para = client_model.state_dict()
    total_time = 0
    total_comm_cost = 0
    epoch_lr = args.lr

    the_number_selection_worker = list()

    for epoch_idx in range(1, 1 + common_config.epoch):
        start_time = time.time()
        # learning rate
        if epoch_idx > 1:
            epoch_lr = max((args.decay_rate * epoch_lr, args.min_lr))

        if common_config.momentum < 0:
            global_optim = optim.SGD(global_model.parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
        else:
            global_optim = optim.SGD(global_model.parameters(), lr=epoch_lr, momentum=common_config.momentum,
                                     nesterov=True, weight_decay=args.weight_decay)

        # selection strategy and batch size configuration for all virtual workers
        selected_ids, bsz_list = control(common_config, active_num, virtual_worker_list)

        # configuration
        communication_parallel(worker_list, epoch_idx, comm, action="send_conf", selected_ids=selected_ids,
                               data=bsz_list)

        # assign data idxes
        communication_parallel(worker_list, epoch_idx, comm, action="assign_data", selected_ids=selected_ids,
                               data=virtual_worker_list)

        # broadcast worker-side model
        communication_parallel(worker_list, epoch_idx, comm, action="send_model", selected_ids=selected_ids,
                               data=global_para)

        global_model.train()
        local_steps = common_config.local_epoch

        for iter_idx in range(local_steps):
            # collect data
            communication_parallel(worker_list, epoch_idx, comm, action="get_para", selected_ids=selected_ids)

            # split training
            train_loss, comm_cost = server_training(args, global_model, global_optim, selected_ids, bsz_list, worker_list)

            # dispatch grad
            communication_parallel(worker_list, epoch_idx, comm, action="send_grad", selected_ids=selected_ids)

            total_comm_cost += comm_cost / 1024 / 1024
            print("\rstep: {} ".format(iter_idx + 1), end='', flush=True)
        print('')
        # get worker-side model
        communication_parallel(worker_list, epoch_idx, comm, action="get_para", selected_ids=selected_ids)

        # aggregate worker-side model
        # global_para = aggregate_model_para(client_model, selected_ids, worker_list, device)
        global_para = aggregate_model_dict(client_model, selected_ids, worker_list, device)
        end_time = time.time()

        test_loss, acc = test(global_model, client_model, test_loader, device)
        logger.info("Epoch: {}, accuracy: {}, test_loss: {}\n".format(epoch_idx, acc, test_loss))
        print("Epoch: {}, accuracy: {}, test_loss: {}".format(epoch_idx, acc, test_loss))

        total_time += end_time - start_time

        total_comm_cost += active_num * client_model_size * 2
        total_bandwith, total_resource = 0, 0

        recorder.add_scalar('Train/time', total_time, epoch_idx)
        recorder.add_scalar('Test/acc', acc, epoch_idx)
        recorder.add_scalar('Test/loss', test_loss, epoch_idx)
        recorder.add_scalar('Test/acc-time', acc, total_time)
        recorder.add_scalar('Train/comm_cost', total_comm_cost, epoch_idx)

        result_out.write(
            '{} {:.2f} {:.2f} {:.2f} {:.4f} {:.4f}'.format(epoch_idx, total_time, total_bandwith, total_resource, acc,
                                                           test_loss))
        result_out.write('\n')

    result_out.close()
    # close socket


def aggregate_model_para(client_model, selected_ids, worker_list, device):
    global_para = torch.nn.utils.parameters_to_vector(client_model.parameters()).detach()
    # weight = 1.0 / len(selected_idxs)
    with torch.no_grad():
        para_delta = torch.zeros_like(global_para).to(device)
        for worker_idx, worker in zip(selected_ids, worker_list):
            para_delta += worker.config.neighbor_paras.to(device)
        para_delta /= len(selected_ids)
    torch.nn.utils.vector_to_parameters(para_delta, client_model.parameters())
    return para_delta


def aggregate_model_dict(client_model, selected_ids, worker_list, device):
    with torch.no_grad():
        local_model_para = []
        for worker_idx, worker in zip(selected_ids, worker_list):
            local_model_para.append(worker.config.neighbor_paras)
        para_delta = copy.deepcopy(local_model_para[0])
        for para in para_delta.keys():
            para_delta[para] = para_delta[para].to(device)
            for i in range(1, len(local_model_para)):
                para_delta[para] += local_model_para[i][para].to(device)
            para_delta[para] = torch.div(para_delta[para], len(local_model_para))
    client_model.load_state_dict(copy.deepcopy(para_delta))
    return para_delta


def communication_parallel(worker_list, epoch_idx, comm, action, selected_ids=[], data=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = []
    for i, worker in enumerate(worker_list):
        if action == "init":
            task = asyncio.ensure_future(worker.send_init_config(comm, epoch_idx))

        elif i < len(selected_ids):
            worker_idx = selected_ids[i]
            if action == "send_conf":
                task = asyncio.ensure_future(worker.send_data((worker_idx, data[worker_idx]), comm, epoch_idx))

            elif action == "assign_data":
                task = asyncio.ensure_future(
                    worker.send_data(data[worker_idx].config.train_data_idxes, comm, epoch_idx))

            elif action == "send_model":
                task = asyncio.ensure_future(worker.send_data(data, comm, epoch_idx))
            elif action == "send_grad":
                task = asyncio.ensure_future(worker.send_data(worker.config.grad_in, comm, epoch_idx))
            elif action == "get_para":
                task = asyncio.ensure_future(worker.get_model(comm, epoch_idx))
        else:
            if action == "send_conf":
                task = asyncio.ensure_future(worker.send_data((-1, -1), comm, epoch_idx))

        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()


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
