from torch.utils.data import DataLoader
from .COCO_Caption import COCO_Captions
from torch.utils.data.dataloader import default_collate


datasets = {
    'coco_caption': COCO_Captions,
}

MAX_WORKERS = 10


def get_dataset(datasetname, cfg, is_train=False, save=False):
    try:
        if save:
            dataset = datasets[datasetname](cfg, is_train=is_train, save=True)
        else:
            dataset = datasets[datasetname](cfg, is_train=is_train)
    except KeyError as e:
        print('*' * 10)
        print('Argument \'dataset\' value: ' + datasetname + ' is not a valid dataset key')
        print('Valid choices for dataset are as follows: ')
        print(list(datasets.keys()))
        print('*' * 10)
        raise e

    return dataset


def get_dataloader(datasetname, cfg, is_train=False, save=False):
    dataset = get_dataset(datasetname, cfg, is_train=is_train, save=save)

    num_workers = cfg.get('n_workers', min(cfg['batchsize'], MAX_WORKERS))

    dataset_loader = DataLoader(dataset=dataset, batch_size=cfg['batchsize'], shuffle=True,
                                num_workers=num_workers, drop_last=is_train, pin_memory=False,
                                collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else default_collate)

    return dataset_loader

