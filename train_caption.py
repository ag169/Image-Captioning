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


from pycocoevalcap.eval import COCOEvalCap


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


def inference(net, inp_tensor, end_index=3, start_index=2, max_token=52):
    assert len(inp_tensor.size()) == 4
    assert inp_tensor.size(0) == 1

    output = [start_index]

    enc_out = net.get_encoder_output(inp_tensor)

    hidden_cell = net.init_hidden_cell(enc_out)

    max_ind = torch.LongTensor([start_index]).to(enc_out.device)

    token_count = 0

    while token_count < max_token:
        net_output, hidden_cell = net.get_decoder_output(max_ind, hidden_cell)

        _, max_ind = torch.max(net_output.squeeze(1), dim=1)

        output.append(max_ind.cpu()[0].item())

        if max_ind == end_index:
            break

        token_count += 1

    return output


def evaluate(model: nn.Module, dataloader, epoch=0, max_token=25):
    time_ = time.time()

    index2token = dataloader.dataset.index2token
    token2index = dataloader.dataset.token2index

    end_token = dataloader.dataset.end_token
    end_token_index = token2index[end_token]

    start_token = dataloader.dataset.start_token
    start_token_index = token2index[start_token]

    pad_token = ''
    pad_token_index = token2index[pad_token]

    ignore_set = {start_token_index, end_token_index, pad_token_index}

    model = model.eval()

    results = list()

    with torch.no_grad():
        for data in dataloader:
            imgs_ip = data[0]
            if args.cuda:
                imgs_ip = imgs_ip.cuda()

            enc_out = model.get_encoder_output(imgs_ip)

            for ii in range(data[0].size(0)):
                output = [start_token_index]

                hidden_cell = net.init_hidden_cell(enc_out[ii][None])

                max_ind = torch.LongTensor([start_token_index]).to(enc_out.device)

                token_count = 0

                while token_count < max_token:
                    net_output, hidden_cell = net.get_decoder_output(max_ind, hidden_cell)

                    _, max_ind = torch.max(net_output.squeeze(1), dim=1)

                    output.append(max_ind.cpu()[0].item())

                    if max_ind == end_token_index:
                        break

                    token_count += 1

                caption_hypothesis = [index2token[x] for x in output if x not in ignore_set]

                hypothesis = ' '.join(caption_hypothesis)
                img_id = data[4][ii]

                result = {
                    "image_id": int(img_id),
                    "caption": hypothesis,
                }

                results.append(result)

    coco_ds = dataloader.dataset.coco_captions
    coco_res = coco_ds.loadRes(results)

    coco_eval = COCOEvalCap(coco_ds, coco_res)

    time_ = (time.time() - time_) / 60
    print('Eval  | Epoch: ' + str(epoch) + '\t' + f'Time: {time_:.2f} mins')

    eval_scores = coco_eval.compute_bleu_cider()

    return eval_scores


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
        cfg = yaml.safe_load(fp)

    log_file = args.cdr + '/log_' + str(datetime.datetime.now()).replace(':', '-').replace(' ', '_') + '.txt'
    print_and_log(log_file)

    print('-' * 50)
    print('Config is as follows:')
    for k, v in cfg.items():
        print(f'{k}: {v}')
    print('-' * 50)

    data_cfg = cfg['dataset']
    datasetname = data_cfg['name']

    data_cfg['val']['separate_captions'] = False

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
    load = cfg.get('load', None)
    epoch_r = -1
    if resume or load:
        if resume:
            load_path = os.path.join(args.cdr, resume)
        else:
            load_path = os.path.join(args.cdr, load)

        state_dict = torch.load(load_path)

        model_state = state_dict['model_state']
        net.load_state_dict(model_state)

        print('Loaded state dict from', load_path)

        if resume:
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
        best_score = evaluate(net, data_loaders['val'], epoch)['CIDEr']
        print('-' * 50)
        tb_writer.add_scalar('val_CIDEr', best_score, 0)
    else:
        best_loss = 1.e6
        best_score = 0

    while epoch < nepochs:
        if freeze_enc_train and epoch >= 0.75 * nepochs:
            print('Unfreezing the image encoder!')
            net.freeze_enc = False
            freeze_enc_train = False

        train_loss = train(net, loss, opt, data_loaders['train'], epoch + 1)
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == nepochs:
            val_score = evaluate(net, data_loaders['val'], epoch + 1)['CIDEr']

        tb_writer.add_scalar('train_loss', train_loss, epoch + 1)
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == nepochs:
            tb_writer.add_scalar('val_CIDEr', val_score, epoch + 1)

        # scheduler.step(val_loss)
        scheduler.step()

        if (epoch + 1) % val_interval == 0 or (epoch + 1) == nepochs:
            if val_score >= best_score:
                best_score = val_score
                print("Saving best checkpoint")
                checkpoint(args.cdr, net, 'best', epoch=epoch + 1)

        checkpoint(args.cdr, net, 'latest', epoch=epoch + 1, _optimizer=opt, _scheduler=scheduler, only_model=False)
        print('-' * 50)

        epoch += 1

    print('Done')
