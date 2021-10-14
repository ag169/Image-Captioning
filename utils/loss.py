import torch.nn as nn


losses = {
    'bce': nn.BCEWithLogitsLoss,
    'ce': nn.CrossEntropyLoss,
    'nll': nn.NLLLoss2d,
}


def get_loss(loss_fn, kwargs=None):
    try:
        if not kwargs:
            return losses[loss_fn]()
        else:
            return losses[loss_fn](**kwargs)
    except KeyError:
        print('Loss function not implemented')
        raise NotImplementedError

