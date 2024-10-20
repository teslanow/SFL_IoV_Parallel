import torch.nn
import torchvision
from torch import optim
from torch.utils.data import DataLoader

from models import AlexNet
from Common.utils import *
from datasets import load_default_transform


def test(model: torch.nn.Module, test_loader, criterion, device):
    model.eval()
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

def local_train():
    set_seed(1234)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transform = load_default_transform("CIFAR10", train=True)
    test_transform = load_default_transform("CIFAR10", train=False)

    train_dataset = torchvision.datasets.CIFAR10("/data/zhongxiangwei/data/CIFAR10", train = True, download = True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10("/data/zhongxiangwei/data/CIFAR10", train = False, download = True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = AlexNet(class_num=10)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train(True)
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss.item())
        test_loss, test_acc = test(model, test_loader, criterion, device)
        print(f"round = {epoch}, test_loss = {test_loss}, acc = {test_acc}")
