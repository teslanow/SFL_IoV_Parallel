import torch.nn
import copy
from typing import List
import numpy as np

def aggregate_model_dict(active_models:List[torch.nn.Module], device, run_samples: List[int]):
    """

    Args:
        active_models:
        device:
        run_samples: 每个model训练的samples总数

    Returns: state_dict

    """
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
                target_dtype = para_delta[para].dtype
                para_delta[para] += (ratios[i] * active_models[i].state_dict()[para].float().to(device)).to(target_dtype)
    return para_delta