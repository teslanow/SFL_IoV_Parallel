from torchvision import models, datasets
import copy
from config import *
import datasets, models
from training_utils import *
from utils import *
import torch.optim as optim
from multiprocessing import Process, Queue, set_start_method


import warnings

# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

def non_iid_partition(ratio, train_class_num, worker_num):
    partition_sizes = np.ones((train_class_num, worker_num)) * ((1 - ratio) / (worker_num-1))

    for i in range(train_class_num):
        partition_sizes[i][i%worker_num]=ratio

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

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes, class_num=train_class_num, labels=labels)
    return train_dataset, test_dataset, train_data_partition, labels

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

def main():
    # set_start_method('spawn', force=True)
    server_device = 'cuda:0'
    args = parse_args()
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    recorder, logger = set_recorder_and_logger(args)
    # init config

    active_num = args.active_num
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

    train_dataset, test_dataset = datasets.load_datasets(args.dataset_type, args.data_path)

    # Create model instance
    _, test_dataset, train_data_partition, labels = partition_data(args.dataset_type,
                                                                   args.data_pattern, args.data_path,
                                                                   worker_num)
    if labels:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False,
                                                  collate_fn=lambda x: datasets.collate_fn(x, labels))
    else:
        test_loader = datasets.create_dataloaders(test_dataset, batch_size=128, shuffle=False)

    # 创建模型
    active_models = []
    for i in range(active_num):
        active_models.append(models.create_model_full(args.dataset_type, args.model_type))

    # 可以使用的cuda devices
    # devices = [f"cuda:{i + 1}" for i in range(7)]

    epoch_lr = args.lr

    glob_model = models.create_model_full(args.dataset_type, args.model_type)

    queue = Queue()

    for epoch_idx in range(args.epoch):
        models_and_args = []
        selected_ids = np.random.choice(range(args.worker_num), size=active_num, replace=False)
        for i in range(len(selected_ids)):
            # 复制模型
            active_models[i].load_state_dict(glob_model.state_dict())
            if args.momentum < 0:
                optimizer = optim.SGD(active_models[i].parameters(), lr=epoch_lr, weight_decay=args.weight_decay)
            else:
                optimizer = optim.SGD(active_models[i].parameters(), lr=epoch_lr, momentum=args.momentum,
                                      nesterov=True, weight_decay=args.weight_decay)

            if labels:
                train_loader = datasets.create_dataloaders_without_helpler(train_dataset, batch_size=int(args.batch_size),
                                                                           selected_idxs=train_data_partition.use(selected_ids[i]), pin_memory=False,
                                                                           drop_last=True,
                                                                           collate_fn=lambda x: datasets.collate_fn(x, labels))
            else:
                train_loader = datasets.create_dataloaders_without_helpler(train_dataset, batch_size=int(args.batch_size),
                                                                           selected_idxs=train_data_partition.use(selected_ids[i]), pin_memory=False,
                                                                           drop_last=True)
            args_current = (active_models[i], train_loader, optimizer, args.local_epoch, server_device, queue)
            models_and_args.append(args_current)
        # 执行模型
        # for args_current in models_and_args:
        #     train_model(args_current)
        # 模型聚合
        result_para_state_dict = aggregate_model_dict(active_models, server_device)
        glob_model.load_state_dict(result_para_state_dict)
        test_loss, acc = test(glob_model, test_loader, server_device)
        logger.info("Epoch: {}, accuracy: {}, test_loss: {}\n".format(epoch_idx, acc, test_loss))
        print("Epoch: {}, accuracy: {}, test_loss: {}".format(epoch_idx, acc, test_loss))
        logger.info("Epoch: {}, accuracy: {}, test_loss: {}\n".format(epoch_idx, acc, test_loss))
        print("Epoch: {}, accuracy: {}, test_loss: {}".format(epoch_idx, acc, test_loss))

        total_time = 0

        total_comm_cost = 0
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

# 定义训练函数
def train_model(args):
    model, dataloader, optimizer, local_epoch,  device, queue = args
    # print(local_epoch)
    for epoch in range(local_epoch):
        model.to(device)
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print(f'Epoch {epoch + 1}/{local_epoch}, Loss: {running_loss / len(dataloader)}')

    queue.put(model)


if __name__ == '__main__':
    main()