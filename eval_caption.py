import argparse
import sys
import copy

import yaml
import os
import torch
from datasets import get_dataloader
from torch.distributions.categorical import Categorical
from models import get_model
from utils.file_ops import make_folder

import torchvision.transforms as tv_t
import numpy as np
import cv2
from tqdm import tqdm

import json
from json import encoder

from pycocoevalcap.eval import COCOEvalCap


encoder.FLOAT_REPR = lambda o: format(o, '.3f')


def inference(net, inp_tensor, end_index=3, start_index=2, max_token=52, random=False):
    assert len(inp_tensor.size()) == 4
    assert inp_tensor.size(0) == 1

    output = [start_index]

    enc_out = net.get_encoder_output(inp_tensor)

    hidden_cell = net.init_hidden_cell(enc_out)

    max_ind = torch.LongTensor([start_index]).to(enc_out.device)

    token_count = 0

    random_temp = 1 if hasattr(net, 'random_temp') else net.random_temp

    while token_count < max_token:
        net_output, hidden_cell = net.get_decoder_output(max_ind, hidden_cell)

        if random:
            net_output_softmax = torch.softmax(net_output.squeeze(1) / random_temp, dim=1)
            m = Categorical(net_output_softmax[0])
            max_ind = m.sample().unsqueeze(0)
        else:
            _, max_ind = torch.max(net_output.squeeze(1), dim=1)

        output.append(max_ind.cpu()[0].item())

        if max_ind == end_index:
            break

        token_count += 1

    return output


def beam_search_inference(net, inp_tensor, beam_size=3, end_index=3, start_index=2, max_token=53, return_all=False):
    assert len(inp_tensor.size()) == 4
    assert inp_tensor.size(0) == 1

    enc_out = net.get_encoder_output(inp_tensor)

    hidden_cell_i = net.init_hidden_cell(enc_out)

    start_input = torch.LongTensor([start_index]).to(enc_out.device)

    net_output, hidden_cell_o = net.get_decoder_output(start_input, hidden_cell_i)

    softmax_score = torch.log_softmax(net_output.squeeze(1), dim=1)
    sorted_score, token_indices = torch.sort(softmax_score, dim=1, descending=True)

    output_list = list()
    for ii in range(beam_size):
        token_ind = token_indices[:, ii]
        score = sorted_score[0, ii]

        output_list.append([[start_input, token_ind], hidden_cell_o, score])

    token_count = 2

    while token_count < max_token:
        token_count += 1

        temp = list()
        c_flag = False

        for seq in output_list:
            token_inds, hidden_cell, score = seq

            if int(token_inds[-1][0]) == end_index:
                temp.append(seq)
            else:
                c_flag = True

                net_output, hidden_cell_o = net.get_decoder_output(token_inds[-1], hidden_cell)

                softmax_score = torch.log_softmax(net_output.squeeze(1), dim=1)
                sorted_score, token_indices = torch.sort(softmax_score, dim=1, descending=True)

                for ii in range(beam_size):
                    token_ind_list = copy.copy(token_inds)
                    token_ind_list.append(token_indices[:, ii])
                    score_new = score + sorted_score[0, ii]
                    temp.append([token_ind_list, hidden_cell_o, score_new])

        if not c_flag:
            break

        temp.sort(key=lambda _x: _x[-1], reverse=True)
        output_list = temp[:beam_size]

    if return_all:
        output = list()

        for beam_op in output_list:
            token_inds, hidden_cell, score = beam_op

            o_list = [int(x_) for x_ in token_inds]

            output.append(o_list)

        return output
    else:
        token_inds = output_list[0][0]
        return [int(x_) for x_ in token_inds]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument('--cdr', type=str, required=True,
                        help='Checkpoint directory')

    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Mixed-Precision Training')

    parser.add_argument('--vis', action='store_true', default=False,
                        help='Enable Visualization Mode')

    parser.add_argument('--suffix', default='latest', type=str)

    parser.add_argument('--beam_size', default='1', type=int)

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

    print('-' * 50)
    print('Config is as follows:')
    for k, v in cfg.items():
        print(f'{k}: {v}')
    print('-' * 50)

    data_cfg = cfg['dataset']
    datasetname = data_cfg['name']

    data_cfg['val']['batchsize'] = 1
    data_cfg['val']['separate_captions'] = False

    data_loader = get_dataloader(datasetname, data_cfg['val'], is_train=False, save=False)

    model_params = cfg['model'].get('params', {})
    model_params['vocab_size'] = data_loader.dataset.num_tokens

    net = get_model(cfg['model']['arch'], params=model_params)

    ckpt_path = os.path.join(args.cdr, f'model_{args.suffix}.pth')
    state_dict = torch.load(ckpt_path)
    model_state = state_dict['model_state']
    net.load_state_dict(model_state)

    if args.cuda:
        net = net.cuda()

    net.eval()

    index2token = data_loader.dataset.index2token
    token2index = data_loader.dataset.token2index

    end_token = data_loader.dataset.end_token
    end_token_index = token2index[end_token]

    start_token = data_loader.dataset.start_token
    start_token_index = token2index[start_token]

    pad_token = ''
    pad_token_index = token2index[pad_token]

    ignore_set = {start_token_index, end_token_index, pad_token_index}

    un_norm = tv_t.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1. / 0.229, 1. / 0.224, 1. / 0.225])

    if args.vis:
        result_dir = os.path.join(args.cdr, 'caption_result')
        make_folder(result_dir)

        captions = list()
        targets = list()

        with torch.no_grad():
            for ii, data in enumerate(data_loader):
                if ii >= 20:
                    break

                x = data[0]

                i_np = un_norm(x[0]).cpu().detach().numpy()

                image = np.transpose(i_np, axes=(1, 2, 0)) * 255
                image = image.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if args.cuda:
                    x = x.cuda()

                if args.beam_size == 1:
                    caption_inds = inference(net, x, end_index=end_token_index, start_index=start_token_index)
                    caption_tokens = [index2token[x] for x in caption_inds]

                else:
                    caption_inds = beam_search_inference(net, x, beam_size=args.beam_size,
                                                         end_index=end_token_index, start_index=start_token_index,
                                                         return_all=True)
                    caption_tokens = [[index2token[x] for x in y] for y in caption_inds]

                target_tokens = [index2token[int(x)] for x in data[1][0]]

                captions.append(caption_tokens)
                targets.append(target_tokens)

                imname = 'img_' + str(ii).zfill(4) + '.jpg'
                cv2.imwrite(os.path.join(result_dir, imname), image)

                print(caption_tokens)
                print(target_tokens)
                print()

                cv2.imshow('img', image)
                cv2.waitKey()

        with open(os.path.join(result_dir, 'results.txt'), 'w') as fp:
            for caption, target in zip(captions, targets):
                fp.writelines(str(caption) + '\n')
                fp.writelines(str(target) + '\n')
                fp.writelines('\n')
    else:
        references = list()
        hypotheses = list()

        results = list()

        with torch.no_grad():
            for data in tqdm(data_loader):
                x = data[0]

                if args.cuda:
                    x = x.cuda()

                if args.beam_size == 1:
                    caption_inds = inference(net, x, end_index=end_token_index, start_index=start_token_index)
                else:
                    caption_inds = beam_search_inference(net, x, beam_size=args.beam_size,
                                                         end_index=end_token_index, start_index=start_token_index)

                caption_hypothesis = [index2token[x] for x in caption_inds if x not in ignore_set]

                hypothesis = ' '.join(caption_hypothesis)
                img_id = data[4]

                result = {
                    "image_id": int(img_id),
                    "caption": hypothesis,
                }

                results.append(result)

        resFile = os.path.join(args.cdr, f'result_beam-size_{args.beam_size}_suffix_{args.suffix}.json')
        with open(resFile, 'w') as fp:
            json.dump(results, fp)

        coco_ds = data_loader.dataset.coco_captions
        cocoRes = coco_ds.loadRes(resFile)

        cocoEval = COCOEvalCap(coco_ds, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        cocoEval.evaluate()

        evaluation_scores = cocoEval.eval

        with open(os.path.join(args.cdr, f'scores__beam-size_{args.beam_size}_suffix_{args.suffix}.txt'), 'w') as fp:
            for k, v in evaluation_scores.items():
                if len(k) < 7:
                    fp.writelines(f'{k}:\t\t{v}\n')
                else:
                    fp.writelines(f'{k}:\t{v}\n')

    print('Done')
