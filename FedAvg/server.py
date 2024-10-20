import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torchvision import models, datasets
import datasets, models
from FedAvgUtil import prepare_running
import torch.optim as optim
from Common.CommDatasets import load_datasets, partition_data
def main():
    args, recorder, logger = prepare_running()
    active_num = args.active_num
    worker_num = args.worker_num
    server_device = args.device

    train_dataset, test_dataset = load_datasets(args.dataset, args.data_path)
    client_datasets = partition_data(train_dataset, worker_num, args.data_pattern)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    

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