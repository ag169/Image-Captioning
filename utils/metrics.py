from .gen_ops import tensor2np
import torch
import numpy as np


class IoU:
    def __init__(self, threshold=0.5, max_val=1.0):
        self.t = threshold
        self.max_val = max_val

    def __call__(self, label, target):
        if isinstance(label, torch.Tensor):
            label = tensor2np(label)
        if isinstance(target, torch.Tensor):
            target = tensor2np(target)

        label_t = label / self.max_val
        label_t = label_t > self.t

        target_t = target / self.max_val
        target_t = target_t > self.t

        intersection = np.logical_and(label_t, target_t)
        union = np.logical_or(label_t, target_t)

        iou = (intersection.sum() + 1.0e-6) / (union.sum() + 1.0e-6)
        return 100. * iou


metric_dict = {
    'iou': IoU
}


def get_metrics(metrics):
    assert isinstance(metrics, dict)
    metric_list = list()
    for k, v in metrics.items():
        if v is None:
            metric = metric_dict[k]()
        elif isinstance(v, list):
            metric = metric_dict[k](*v)
        else:
            metric = metric_dict[k](**v)
        metric_list.append(metric)

    return metric_list
