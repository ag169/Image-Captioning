import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

from pycocotools.coco import COCO
import nltk

import torchvision.transforms as tv_t
from torchvision.transforms.functional import InterpolationMode
from utils.img_transforms import Square

from PIL import Image
import numpy as np
import cv2

import random
import pickle
from collections import OrderedDict


DATASET_ROOT = 'C:\\Users\\megaa\\Python Projects\\Datasets\\MSCOCO_2014'


def collate_fn(batch, padding_value=0):
    collated = list()

    lengths = list()

    batch.sort(key=lambda x: len(x[1]), reverse=True)

    for i in range(len(batch[0])):
        c_x = [x[i] for x in batch]

        if i == 1:
            lengths = [len(x) for x in c_x]
            c_x = pad_sequence(c_x, batch_first=True, padding_value=padding_value)
        else:
            c_x = default_collate(c_x)

        collated.append(c_x)

    collated.append(lengths)

    return tuple(collated)


def ann2lists(captions_ann, separate_captions=True):
    if separate_captions:
        anns_list = [captions_ann.loadAnns(x)[0] for x in captions_ann.anns]

        captions_list = [y['caption'] for y in anns_list]

        img_names_list = [captions_ann.loadImgs(y['image_id'])[0]['file_name'] for y in anns_list]

        img_ids = [y['image_id'] for y in anns_list]
    else:
        img_ids = list(captions_ann.imgs.keys())

        img_names_list = [x['file_name'] for x in captions_ann.loadImgs(img_ids)]

        captions_list = [[y['caption'] for y in captions_ann.loadAnns(captions_ann.getAnnIds(imgIds=x))]
                         for x in img_ids]

        img_ids = [x['id'] for x in captions_ann.loadImgs(img_ids)]

    return img_names_list, captions_list, separate_captions, img_ids


class COCO_Captions(Dataset):
    def __init__(self, cfg=None, is_train=False, save=False):
        if cfg is None:
            cfg = dict()

        self.batch_size = cfg.get('batchsize', 1)

        self.in_channels = 3

        self.split = 'train' if is_train else 'val'

        self.img_dir = os.path.join(DATASET_ROOT, f'{self.split}2014')
        captions_ann_path = os.path.join(DATASET_ROOT, 'annotations', f'captions_{self.split}2014.json')

        self.coco_captions = COCO(captions_ann_path)

        self.imgs, self.captions, self.separate_captions, self.image_ids \
            = ann2lists(self.coco_captions, separate_captions=cfg.get('separate_captions', True))

        self.imgsize = cfg.get('imgsize', 256)

        t_list = [
            Square(size=int(self.imgsize * 1.1), stretch=False, interpolation=InterpolationMode.BILINEAR),

            # TODO: Try with these augmentations (and come up with more if needed)
            # tv_t.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
            # tv_t.RandomResizedCrop(size=self.imgsize, scale=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR),
            # tv_t.AutoAugment(interpolation=InterpolationMode.BILINEAR),

            tv_t.RandomCrop(size=self.imgsize),
            tv_t.RandomHorizontalFlip(),
            tv_t.ToTensor(),
            tv_t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ] if is_train else [
            Square(size=self.imgsize, stretch=False, interpolation=InterpolationMode.BILINEAR),
            tv_t.ToTensor(),
            tv_t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        self.img_transforms = tv_t.Compose(t_list)

        self.token_count_thresh = cfg.get('token_count_thresh', 10)

        token_path = os.path.join(DATASET_ROOT, 'annotations', 'captions_tokens_count.pkl')
        with open(token_path, 'rb') as fp:
            token_count = OrderedDict(pickle.load(fp))

        self.unknown_token = '</UNK>'
        self.start_token = '</START>'
        self.end_token = '</END>'

        token_dict = OrderedDict({
            '': -1,
            self.unknown_token: -1,
            self.start_token: -1,
            self.end_token: -1,
        })

        token_dict.update(
            OrderedDict([(k, token_count[k]) for k in sorted(token_count.keys())
                         if token_count[k] > self.token_count_thresh])
        )

        self.token_count_dict = token_dict

        self.token2index = OrderedDict(
            [(tkn, ii) for (ii, tkn) in enumerate(self.token_count_dict.keys())]
        )

        self.index2token = OrderedDict(
            [x for x in enumerate(self.token_count_dict.keys())]
        )

        self.num_tokens = len(self.index2token.keys())

        self.r1 = random.Random()

        self.collate_fn = collate_fn

        self.max_len = 50

    def __len__(self):
        # Set length to 100 for debugging
        # return 100
        return len(self.imgs)

    def __getitem__(self, index):
        img_name = self.imgs[index]

        if self.separate_captions:
            caption = self.captions[index]
        else:
            caption = random.choice(self.captions[index])

        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        image_tensor = self.img_transforms(image)

        tokens = nltk.tokenize.word_tokenize(caption.lower())

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        index_list = list()

        index_list.append(self.token2index['</START>'])

        for token in tokens:
            if token in self.token2index:
                index_list.append(self.token2index[token])
            else:
                index_list.append(self.token2index["</UNK>"])

        index_list.append(self.token2index['</END>'])

        tokenized_word_tensor = torch.LongTensor(index_list)

        ret_list = [image_tensor, tokenized_word_tensor, caption]

        if not self.separate_captions:
            ret_list.append(self.captions[index])
            ret_list.append(self.image_ids[index])

        return ret_list


if __name__ == '__main__':
    coco_ds = COCO_Captions(is_train=False)

    num_imgs = 10

    un_norm = tv_t.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1./0.229, 1./0.224, 1./0.225])

    for ii in range(num_imgs):
        index = random.randint(0, len(coco_ds) - 1)
        i_tensor, cap_t, caption = coco_ds.__getitem__(index)

        i_np = un_norm(i_tensor).cpu().detach().numpy()

        image = np.transpose(i_np, axes=(1, 2, 0)) * 255
        image = image.astype(np.uint8)

        token_caption = [coco_ds.index2token[int(x)] for x in cap_t]

        print(caption)
        print(token_caption)

        cv2.imshow('img', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

