import argparse
import sys
import cv2
import yaml
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
from models import get_model
from datasets import get_caption_loader as get_dataloader
from utils.loss import get_loss
import torchvision.transforms as tv_t
import numpy as np
import time


def forward_with_loss(model: nn.Module, criterion: nn.Module, ip: torch.Tensor, captions, lengths, targets,
                      is_train=True, cuda=True):
    if cuda:
        ip = ip.cuda()
        captions = captions.cuda()
        targets = targets.cuda()

    if is_train:
        model.train()
        output = model(ip, captions, lengths)
        # output = output.permute(0, 2, 1)
        _loss = criterion(output, targets)
        # _loss = criterion(output.view(-1, output.size(2)), captions.view(-1))
    else:
        model.eval()
        with torch.no_grad():
            output = model(ip, captions, lengths)
            # output = output.permute(0, 2, 1)
            _loss = criterion(output, targets)
            # _loss = criterion(output.view(-1, output.size(2)), captions.view(-1))

    return output, _loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument('--cdr', type=str, required=True,
                        help='Checkpoint directory')

    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Mixed-Precision Training')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    if not os.path.isdir(args.cdr):
        print(f'Directory {args.cdr} not found!')
        sys.exit()

    cfg_path = os.path.join(args.cdr, 'config.yml')

    if not os.path.isfile(cfg_path):
        print(f'Cfg file not found at {cfg_path}!')
        sys.exit()

    with open(cfg_path) as fp:
        cfg = yaml.load(fp)

    print('-' * 50)
    print('Config is as follows:')
    for k, v in cfg.items():
        print(f'{k}: {v}')
    print('-' * 50)

    data_cfg = cfg['dataset']
    datasetname = data_cfg['name']

    data_loaders = {
        'train': get_dataloader(datasetname, data_cfg['train'], is_train=True, save=False),
        'val': get_dataloader(datasetname, data_cfg['val'], is_train=False, save=False),
    }

    # loss = get_loss(cfg['loss'], kwargs={'ignore_index': 0})
    loss = get_loss(cfg['loss'])

    model_params = cfg['model'].get('params', {})
    model_params['vocab_size'] = data_loaders['train'].dataset.num_tokens

    net = get_model(cfg['model']['arch'], params=model_params)

    if cfg.get('freeze_enc', False):
        print('Freezing image encoder for a while!')
        net.freeze_enc = True

    # Add config based loading for arguments

    if args.cuda:
        net = net.cuda()
        loss = loss.cuda()

    # opt = optim.SGD(net.parameters(), lr=cfg['lr'], momentum=0.9, nesterov=True,
    #                 weight_decay=cfg.get('weight_decay', 1e-5))
    opt = optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 1e-5))

    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

    if args.amp:
        # net, opt = apex.amp.initialize(net, opt, opt_level="O1")
        amp_scaler = torch.cuda.amp.GradScaler()

    resume = cfg.get('resume', None)
    epoch_r = -1
    if resume:
        state_dict = torch.load(os.path.join(args.cdr, resume))
        model_state = state_dict['model_state']
        net.load_state_dict(model_state)

        epoch_r = state_dict.get('epoch', -1)

        try:
            opt_state = state_dict['optimizer']
            opt.load_state_dict(opt_state)
        except KeyError or RuntimeError:
            print('Can\'t load optimizer dict')

        try:
            scheduler_state = state_dict['scheduler']
            scheduler.load_state_dict(scheduler_state)
        except KeyError or RuntimeError:
            print('Can\'t load scheduler')

    val_interval = cfg.get('val_interval', 1)

    nepochs = cfg.get('nepochs', 100)
    if epoch_r < 0:
        epoch = 0
    else:
        epoch = epoch_r

    print('-' * 50)
    print('Start Training!')
    print('-' * 50)

    data = next(iter(data_loaders['train']))

    while epoch < 50:
        # data = next(iter(data_loaders['train']))
        lengths = [x - 1 for x in data[-1]]

        with torch.cuda.amp.autocast(enabled=args.amp):
            targets = pack_padded_sequence(data[1][:, 1:], lengths, batch_first=True)[0]

            output, _loss = forward_with_loss(model=net, criterion=loss, ip=data[0], captions=data[1],
                                              targets=targets, lengths=lengths)

        if args.amp:
            # with apex.amp.scale_loss(_loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            amp_scaler.scale(_loss).backward()
            # torch.nn.utils.clip_grad_norm_(net.lstm .parameters(), 100.0)
            amp_scaler.step(opt)
            amp_scaler.update()
        else:
            _loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.lstm.parameters(), 100.0)
            opt.step()

        print('Train | Epoch: ' + str(epoch) + '\t' + f'Loss: {_loss.item():.5f}\t')

        print('-'*50)

        epoch += 1

    net.eval()

    index2token = data_loaders['train'].dataset.index2token
    token2index = data_loaders['train'].dataset.token2index

    end_token = data_loaders['train'].dataset.end_token
    end_token_index = token2index[end_token]

    un_norm = tv_t.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1. / 0.229, 1. / 0.224, 1. / 0.225])

    with torch.no_grad():
        for ii in range(data[0].size(0)):
            x = data[0][ii][None, ...]

            i_np = un_norm(x[0]).cpu().detach().numpy()

            image = np.transpose(i_np, axes=(1, 2, 0)) * 255
            image = image.astype(np.uint8)

            caption_inds = net.inference(x.cuda(), end_index=end_token_index)

            caption_tokens = [index2token[x] for x in caption_inds]
            target_tokens = [index2token[int(x)] for x in data[1][ii]]

            print(caption_tokens)
            print(target_tokens)
            print()

            cv2.imshow('img', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey()

    print('Done')
