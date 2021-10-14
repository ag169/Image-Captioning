import argparse
import datetime
import sys
import yaml
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim
import tensorboardX
from models import get_model
from datasets import get_dataloader
from utils.loss import get_loss
from utils.file_ops import print_and_log
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
        _loss = criterion(output, targets)
    else:
        model.eval()
        with torch.no_grad():
            output = model(ip, captions, lengths)
            _loss = criterion(output, targets)

    return output, _loss


def train(model: nn.Module, criterion: nn.Module, optimizer, dataloader, epoch=0, print_interval=1000):
    time_ = time.time()

    model = model.train()

    total_loss = 0
    count = 0

    print_loss = 0
    print_count = 0
    print_time = time_

    for ii, data in enumerate(dataloader):
        optimizer.zero_grad()

        images = data[0]
        captions = data[1]
        lengths = [x-1 for x in data[-1]]
        targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]

        if args.amp:
            with torch.cuda.amp.autocast():
                output, _loss = forward_with_loss(model, criterion, ip=images, captions=captions, lengths=lengths,
                                                  targets=targets, is_train=True, cuda=args.cuda)
        else:
            output, _loss = forward_with_loss(model, criterion, ip=images, captions=captions, lengths=lengths,
                                              targets=targets, is_train=True, cuda=args.cuda)

        # TODO: Add gradient clipping for the LSTM layer
        if args.amp:
            # with apex.amp.scale_loss(_loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            amp_scaler.scale(_loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            _loss.backward()
            optimizer.step()

        print_loss += float(_loss)
        print_count += 1

        total_loss += float(_loss)
        count += 1

        if print_count == print_interval:
            print_time = (time.time() - print_time) / 60
            print_loss /= print_count

            print('Train | Epoch: ' + str(
                epoch) + f'\tAvg Loss: {print_loss:.5f}\tTime: {print_time:.2f} mins\tNo. of steps: {count}')

            print_loss = 0
            print_count = 0
            print_time = time.time()

    if count >= print_interval:
        print()

    total_loss /= count
    time_ = (time.time() - time_) / 60

    print('Train | Epoch: ' + str(epoch) + f'\tAvg Loss: {total_loss:.5f}\tTime: {time_:.2f} mins\tNo. of steps: '
                                           f'{count}')

    return total_loss


def evaluate(model: nn.Module, criterion: nn.Module, dataloader, epoch=0):
    time_ = time.time()

    model = model.eval()

    total_loss = 0
    count = 0

    for data in dataloader:
        images = data[0]
        captions = data[1]
        lengths = [x - 1 for x in data[-1]]
        targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]

        if args.amp:
            with torch.cuda.amp.autocast():
                output, _loss = forward_with_loss(model, criterion, ip=images, captions=captions, lengths=lengths,
                                                  targets=targets, is_train=False, cuda=args.cuda)
        else:
            output, _loss = forward_with_loss(model, criterion, ip=images, captions=captions, lengths=lengths,
                                              targets=targets, is_train=False, cuda=args.cuda)

        total_loss += float(_loss) * images.size(0)
        count += images.size(0)

    total_loss /= count
    time_ = (time.time() - time_) / 60

    print('Eval  | Epoch: ' + str(epoch) + '\t' + f'Loss: {total_loss:.5f}\t' + f'Time: {time_:.2f} mins')

    return total_loss


def checkpoint(checkpointdir, model: nn.Module,  suffix='best', epoch=-1, _optimizer=None, _scheduler=None,
               only_model=True):
    state_dict = dict()

    state_dict['model_state'] = model.state_dict()
    state_dict['epoch'] = epoch

    if not only_model:
        if _optimizer is not None:
            state_dict['optimizer'] = _optimizer.state_dict()
        if _scheduler is not None:
            state_dict['scheduler'] = _scheduler.state_dict()

    torch.save(state_dict, checkpointdir + '/model_' + suffix + '.pth')


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

    log_file = args.cdr + '/log_' + str(datetime.datetime.now()).replace(':', '-').replace(' ', '_') + '.txt'
    print_and_log(log_file)

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

    freeze_enc_train = cfg.get('freeze_enc', False)

    if freeze_enc_train:
        print('Freezing image encoder for a while!')
        net.freeze_enc = True

    if args.cuda:
        net = net.cuda()
        loss = loss.cuda()

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

    tb_writer = tensorboardX.SummaryWriter(args.cdr)

    val_interval = cfg.get('val_interval', 1)

    nepochs = cfg.get('nepochs', 100)
    if epoch_r < 0:
        epoch = 0
    else:
        epoch = epoch_r

    print('-' * 50)
    print('Start Training!')
    early_eval = cfg.get('early_eval', True)
    print('-' * 50)
    if early_eval:
        best_loss = evaluate(net, loss, data_loaders['val'], epoch)
        print('-' * 50)
        tb_writer.add_scalar('val_loss', best_loss, 0)
    else:
        best_loss = 1.e6

    while epoch < nepochs:
        if freeze_enc_train and epoch >= 0.75 * nepochs:
            print('Unfreezing the image encoder!')
            net.freeze_enc = False
            freeze_enc_train = False

        train_loss = train(net, loss, opt, data_loaders['train'], epoch + 1)
        tb_writer.add_scalar('train_loss', train_loss, epoch + 1)

        if (epoch + 1) % val_interval == 0:
            val_loss = evaluate(net, loss, data_loaders['val'], epoch + 1)
            tb_writer.add_scalar('val_loss', val_loss, epoch + 1)

            if val_loss < best_loss:
                best_loss = val_loss
                print("Saving best_loss checkpoint")
                checkpoint(args.cdr, net, 'best', epoch=epoch + 1)

        # scheduler.step(val_loss)
        scheduler.step()

        checkpoint(args.cdr, net, 'latest', epoch=epoch + 1, _optimizer=opt, _scheduler=scheduler, only_model=False)
        print('-' * 50)

        epoch += 1

    print('Done')
