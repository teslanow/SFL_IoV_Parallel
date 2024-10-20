from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet34
from ptflops import get_model_complexity_info

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

def create_SL_resnet(model: nn.Module, split_point:int, class_num:int):
    """
    每个模型都要单独设计
    Args:
        model:
        split_point:在第几层之后分割
        class_num:
        pretrained:
    Returns: bottom model, top model
    """
    all_indivisible_module = [model.conv1, model.bn1, model.relu, model.maxpool]
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for layer in layers:
        for child in layer.children():
            all_indivisible_module.append(child)
    all_indivisible_module.append(model.avgpool)
    # 更改为新的class_num
    all_indivisible_module.append(nn.Linear(model.fc.in_features, class_num))
    # all_indivisible_module.append(model.fc)
    if split_point <= 0 or split_point >= len(all_indivisible_module) - 2:
        raise Exception(f"超出划分点，共有{len(all_indivisible_module)}个不可分的module")
    return ResnetBottom(all_indivisible_module[:split_point]), ResnetTop(all_indivisible_module[split_point:])


def create_model_instance_SL(model_type, create_args:dict):
    if model_type == 'Resnet34':
        split_point, class_num, pretrained = create_args['split_point'], create_args['class_num'], create_args['pretrained']
        model = resnet34(pretrained=pretrained)
        bottom_model, top_model = create_SL_resnet(model, split_point, class_num)
        return bottom_model, top_model
    else:
        raise NotImplementedError


def create_model_full(model_type, create_args):
    if model_type == 'Resnet34':
        class_num, pretrained = create_args
        model = resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, class_num)
        return model
    else:
        raise NotImplementedError


def model_density_per_layer(model: torch.nn.Module, input_res: Tuple[int, int, int]):
    """
    返回model前向需要的mac数，unit in multiply-add-num
    返回参数数量， unit in 个
    Args:
        model:
        input_res: 每个batch的输入resolution，例如(3, 32, 32)

    Returns:

    """
    macs, params = get_model_complexity_info(model, input_res, as_strings=False, backend='pytorch',
                                             print_per_layer_stat=False, verbose=False)
    return macs, params

if __name__ == "__main__":
    b, t = create_model_instance_SL('Resnet34', {'split_point': 5, 'class_num': 100, 'pretrained': True})
    m, p = model_density_per_layer(b, (3, 224, 224))
    print(b)
    print(t)
    print(m)
    print(p)
