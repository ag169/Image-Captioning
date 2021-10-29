import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.avg = size_average

    def forward(self, outputs, targets):
        log_pt = -self.ce_loss(outputs, targets)

        pt = log_pt.exp()

        focal_loss = -1 * torch.pow(1. - pt, self.gamma) * log_pt

        if self.avg:
            return focal_loss.mean()

        return focal_loss


losses = {
    'bce': nn.BCEWithLogitsLoss,
    'ce': nn.CrossEntropyLoss,
    'nll': nn.NLLLoss,
    'focal': FocalLoss,
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


if __name__ == '__main__':
    batch = 32
    num_class = 1000
    L = 150

    loss_criterion = get_loss('focal')

    net_op = torch.rand(size=[batch, num_class, L], dtype=torch.float)
    label = torch.randint(low=1, high=num_class, size=[batch, L]).to(torch.long)

    loss = loss_criterion(net_op, label)

    print('Done')

