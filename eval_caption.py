import argparse
import sys

import yaml
import os
import torch
from datasets import get_dataloader
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

                caption_inds = net.inference(x, end_index=end_token_index)

                caption_tokens = [index2token[x] for x in caption_inds]
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

                caption_inds = net.inference(x, end_index=end_token_index)

                if isinstance(caption_inds[0], int):
                    caption_inds_single = [x for x in caption_inds if x not in ignore_set]
                else:
                    # TODO: Use the confidence for each caption and send the best caption

                    '''
                    caption_inds = [[x for x in y if x not in ignore_set] for y in caption_inds]
                    caption_inds_single = caption_inds[0]
                    '''

                    raise NotImplementedError

                caption_hypothesis = [index2token[x] for x in caption_inds_single]

                hypothesis = ' '.join(caption_hypothesis)
                img_id = data[4]

                result = {
                    "image_id": int(img_id),
                    "caption": hypothesis,
                }

                results.append(result)

        resFile = os.path.join(args.cdr, f'result_{args.suffix}.json')
        with open(resFile, 'w') as fp:
            json.dump(results, fp)

        coco_ds = data_loader.dataset.coco_captions
        cocoRes = coco_ds.loadRes(resFile)

        cocoEval = COCOEvalCap(coco_ds, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()

        cocoEval.evaluate()

        evaluation_scores = cocoEval.eval

        with open(os.path.join(args.cdr, f'scores_{args.suffix}.txt'), 'w') as fp:
            for k, v in evaluation_scores.items():
                if len(k) < 7:
                    fp.writelines(f'{k}:\t\t{v}\n')
                else:
                    fp.writelines(f'{k}:\t{v}\n')

    print('Done')
