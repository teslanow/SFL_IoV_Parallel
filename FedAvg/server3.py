import torch.nn
from torchvision import models, datasets
import copy
from config import *
import models
from Common.utils import *
import torch.optim as optim


def aggregate_model_dict(active_models:List[torch.nn.Module], device, run_samples: List[int]):
    # return state_dict
    with torch.no_grad():
        para_delta = copy.deepcopy(active_models[0].state_dict())
        keys = para_delta.keys()
        for para in keys:
            para_delta[para].zero_()
            para_delta[para].to(device)
        total_samples = sum(run_samples)
        ratios = (np.array(run_samples) / total_samples).tolist()
        for para in keys:
            for i in range(0, len(active_models)):
                para_delta[para] += (ratios[i] * active_models[i].state_dict()[para].to(device))
    return para_delta

def main():
    args = parse_args()
    server_device = args.device
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
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

    train_dataset, test_dataset, train_data_partition, labels = partition_data(args.dataset_type, args.data_pattern, args.data_path, worker_num)
    test_loader = datasets.create_dataloaders_fedavg(test_dataset, batch_size=256, shuffle=False)

    # 创建模型
    active_models = []
    for i in range(active_num):
        active_models.append(models.create_model_full(args.dataset_type, args.model_type, 100))

    epoch_lr = args.lr
    glob_model = models.create_model_full(args.dataset_type, args.model_type, 100)
    criterion = torch.nn.CrossEntropyLoss()

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
            train_loader = datasets.create_dataloaders_fedavg(train_dataset, batch_size=int(args.batch_size),
                                                                           selected_idxs=train_data_partition.use(selected_ids[i]), pin_memory=False,
                                                                           drop_last=True)
            args_current = (active_models[i], train_loader, optimizer, args.local_epoch, server_device, criterion)
            models_and_args.append(args_current)
        run_samples = []
        # 执行模型
        for i, args_current in enumerate(models_and_args):
            run_total_batch_size = train_model(args_current)
            print(f'\rcurrent: {i}', end='')
            run_samples.append(run_total_batch_size)
        print('')
        # 模型聚合
        result_para_state_dict = aggregate_model_dict(active_models, server_device, run_samples)
        glob_model.load_state_dict(result_para_state_dict)
        test_loss, acc = evaluate(glob_model, test_loader, server_device, criterion)
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
    model, dataloader, optimizer, local_epoch,  device, criterion = args
    model.to(device)
    model.train()
    run_total_batch_size = 0
    # print(local_epoch)
    for epoch in range(local_epoch):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_total_batch_size += inputs.size(0)
            # print(loss)
    return run_total_batch_size

def evaluate(model: torch.nn.Module, test_loader, device, criterion):
    model.eval()
    model.to(device)
    test_loss = 0.0
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    avg_test_loss = test_loss / total
    avg_test_acc = correct / total
    return avg_test_loss, avg_test_acc


if __name__ == '__main__':
    main()