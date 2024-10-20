import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

class ResnetBottom(nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x

class ResnetTop(nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        for module in self.module_list[:-1]:
            x = module(x)
        x = torch.flatten(x, 1)
        x = self.module_list[-1](x)
        return x

def create_SL_resnet(split_point:int, class_num:int, pretrained: bool):
    """
    每个模型都要单独设计
    Args:
        model:
        split_point:在第几层之后分割
        class_num:
        pretrained:

    Returns: bottom model, top model

    """
    model = resnet34(pretrained=pretrained)
    all_indivisible_module = [model.conv1, model.bn1, model.relu, model.maxpool]
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for layer in layers:
        for child in layer.children():
            all_indivisible_module.append(child)
    all_indivisible_module.append(model.avgpool)
    all_indivisible_module.append(model.fc)
    if split_point <= 0 or split_point >= len(all_indivisible_module) - 2:
        raise Exception(f"超出划分点，共有{len(all_indivisible_module)}个不可分的module")
    return ResnetBottom(all_indivisible_module[:split_point]), ResnetTop(all_indivisible_module[split_point:])


# 加载预训练的 ResNet-34 模型
bottom_model, top_model = create_SL_resnet(5, 10, True)
# model.eval()  # 设置为评估模式

# 加载 CIFAR-100 数据集
transform = transforms.Compose([
    transforms.Resize(224),  # ResNet-34 需要 224x224 的输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

cifar100_dataset = CIFAR100(root='/data/zhongxiangwei/data/CIFAR100/', train=False, download=True, transform=transform)
cifar100_loader = DataLoader(cifar100_dataset, batch_size=1, shuffle=False)

# 获取第一张图片
image, label = next(iter(cifar100_loader))

# 使用模型进行预测
with torch.no_grad():
    bottom_model.eval()
    top_model.eval()
    x = bottom_model(image)
    output = top_model(x)

# 获取预测结果
_, predicted_idx = torch.max(output, 1)

model2 = resnet34(pretrained=True)
with torch.no_grad():
    bottom_model.eval()
    top_model.eval()
    x = bottom_model(image)
    output = top_model(x)
_, predicted_idx2 = torch.max(output, 1)

print(predicted_idx)
print(predicted_idx2)
