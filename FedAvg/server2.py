import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.distributions.dirichlet import Dirichlet
from collections import OrderedDict
import copy

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# 加载数据集
train_dataset = datasets.CIFAR100(root='/data/zhongxiangwei/data/CIFAR100', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='/data/zhongxiangwei/data/CIFAR100', train=False, download=True, transform=transform)

# 数据集大小
train_size = len(train_dataset)
num_clients = 10

# 使用 Dirichlet 分布划分数据集
dirichlet_dist = Dirichlet(torch.tensor([1.0] * num_clients))
client_data_sizes = (dirichlet_dist.sample() * train_size).int().tolist()

# 处理client_data_sizes使得正好加起来的值等于len(train_dataset)
diff = int(abs(sum(client_data_sizes) - len(train_dataset)))
if diff > 0:
    t = math.ceil(diff / num_clients)
    i = 0
    while diff > 0:
        add_num = min(diff, t)
        client_data_sizes[i] += add_num
        diff -= add_num
        i += 1
client_datasets = random_split(train_dataset, client_data_sizes)

# 数据加载器
train_loaders = [DataLoader(ds, batch_size=64, shuffle=True) for ds in client_datasets]
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 使用 ResNet-18 模型
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 100)

# 如果可用，将模型移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def train_fedavg(model, train_loaders, criterion, optimizer, num_epochs, num_clients, device):
    global_model = copy.deepcopy(model)
    global_weights = global_model.state_dict()

    for epoch in range(num_epochs):
        local_weights = []

        for client_idx in range(num_clients):
            local_model = copy.deepcopy(global_model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            local_model.train()

            for inputs, labels in train_loaders[client_idx]:
                inputs, labels = inputs.to(device), labels.to(device)

                local_optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                local_optimizer.step()

            local_weights.append(copy.deepcopy(local_model.state_dict()))

        # 聚合权重
        avg_weights = OrderedDict()
        for key in global_weights.keys():
            avg_weights[key] = torch.stack([local_weights[i][key].float() for i in range(num_clients)], 0).mean(0)

        global_model.load_state_dict(avg_weights)
        global_weights = global_model.state_dict()

        # 测试
        test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
        print('Round {}, Test Loss {}, Test Acc {}%'.format(epoch, test_loss, test_acc))

    return global_model

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

num_epochs = 100
num_clients = 10

# 训练 FedAvg
global_model = train_fedavg(model, train_loaders, criterion, optimizer, num_epochs, num_clients, device)

# 评估模型
test_loss, test_acc = evaluate(global_model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')