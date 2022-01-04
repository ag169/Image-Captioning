import argparse
import sys

import yaml
import os
import torch
from datasets import get_dataloader
from models import get_model

import torchvision.transforms as tv_t
from PIL import Image
from utils.img_transforms import Square

import matplotlib.pyplot as plt

from eval_caption import inference, beam_search_inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset Arguments
    parser.add_argument('--cdr', type=str, default='checkpoints\\c11_eb4_gru_attn2',
                        help='Checkpoint directory')

    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Mixed-Precision Mode')

    parser.add_argument('--img_path', type=str, default='test.jpg')

    parser.add_argument('--suffix', default='best', type=str)

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

    if not os.path.isfile(args.img_path):
        print(f'Image file not found at {args.img_path}')
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

    img_transform = tv_t.Compose([
            Square(size=data_cfg['val']['imgsize'], stretch=False),
            tv_t.ToTensor(),
            tv_t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # un_norm = tv_t.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    #                          std=[1. / 0.229, 1. / 0.224, 1. / 0.225])

    img = Image.open(args.img_path).convert('RGB')

    img_tensor = img_transform(img)[None]

    if args.cuda:
        img_tensor = img_tensor.cuda()

    captions = list()

    if args.beam_size == 1:
        caption_inds = inference(net, img_tensor, end_index=end_token_index, start_index=start_token_index)
        caption_tokens = [index2token[x] for x in caption_inds]

        captions.append(' '.join(caption_tokens))
    else:
        caption_inds = beam_search_inference(net, img_tensor, beam_size=args.beam_size,
                                             end_index=end_token_index, start_index=start_token_index,
                                             return_all=True)
        caption_tokens = [[index2token[x] for x in y] for y in caption_inds]

        captions.extend(' '.join(x) for x in caption_tokens)

    print('Generated caption(s):')
    for caption in captions:
        print(caption)

    plt.imshow(img)
    plt.title(captions[0])
    plt.show()

    print('Done')

