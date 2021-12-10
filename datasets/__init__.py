from torch.utils.data import DataLoader
from .COCO_Caption import COCO_Captions
from .COCO_Caption_Synonyms import COCO_Captions_Synonyms
from torch.utils.data.dataloader import default_collate


datasets = {
    'coco_caption': COCO_Captions,
    'coco_caption_synonyms': COCO_Captions_Synonyms
}

MAX_WORKERS = 10


def get_dataset(datasetname, cfg, is_train=False, save=False, load_embed=False):
    try:
        if save:
            dataset = datasets[datasetname](cfg, is_train=is_train, save=True, load_embed=load_embed)
        else:
            dataset = datasets[datasetname](cfg, is_train=is_train, load_embed=load_embed)
    except KeyError as e:
        print('*' * 10)
        print('Argument \'dataset\' value: ' + datasetname + ' is not a valid dataset key')
        print('Valid choices for dataset are as follows: ')
        print(list(datasets.keys()))
        print('*' * 10)
        raise e

    return dataset


def get_dataloader(datasetname, cfg, is_train=False, save=False, load_embed=False):
    dataset = get_dataset(datasetname, cfg, is_train=is_train, save=save, load_embed=load_embed)

    num_workers = cfg.get('n_workers', min(cfg['batchsize'], MAX_WORKERS))

    dataset_loader = DataLoader(dataset=dataset, batch_size=cfg['batchsize'], shuffle=True,
                                num_workers=num_workers, drop_last=is_train, pin_memory=False,
                                collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else default_collate)

    return dataset_loader

