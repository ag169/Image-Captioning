import os
import sys
import pathlib
import random
import cv2

from os.path import join as p_join


class DoubleWrite:
    def __init__(self, out1, out2):
        self.out1 = out1
        self.out2 = out2

    def write(self, *args, **kwargs):
        self.out1.write(*args, **kwargs)
        self.out2.write(*args, **kwargs)

    def flush(self, *args, **kwargs):
        self.out1.flush(*args, **kwargs)
        self.out2.flush(*args, **kwargs)


def print_and_log(logfile):
    class FileAndPrint:
        def __init__(self, f, out):
            self.file = f
            self.out = out

        def write(self, *args, **kwargs):
            self.out.write(*args, **kwargs)
            f = open(self.file, "a")
            f.write(*args, **kwargs)
            f.close()

        def flush(self, *args, **kwargs):
            self.out.flush(*args, **kwargs)

    sys.stdout = FileAndPrint(logfile, sys.stdout)


def check_file(path):
    return os.path.isfile(path)


def check_folder(path):
    return os.path.isdir(path)


def make_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def remove_file(path):
    if check_file(path):
        os.remove(path)


def read_file(path):
    if not check_file(path):
        raise Exception('File pointed to by \'', path, '\' not found')
    return open(path, 'r')


def text2list(file):
    pl = []
    for f in file.readlines():
        pl.append(f.strip())
    return pl


def text_file(path, save_old=True):
    # file1 = open('', 'w')
    if path[-4:] != '.txt':
        path = path + '.txt'

    print('Creating text file: ', path)

    fname = os.path.splitext(path)[0]

    if save_old and check_file(path):
        new_path = fname + '_old.txt'
        remove_file(new_path)
        os.rename(path, new_path)
        print('Moved file', path, ' to ', new_path)

    file = open(path, 'w')
    return file


def train_val_split(image_dir, gt_dir, file_root=None, ratio=0.8, save_old=False):
    if file_root is None:
        file_root = image_dir + '/..'
        file_root = os.path.abspath(file_root)

    image_root = pathlib.Path(image_dir)
    all_image_paths = list(image_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    gt_root = pathlib.Path(gt_dir)
    all_gt_paths = list(gt_root.glob('*'))
    all_gt_paths = [str(path) for path in all_gt_paths]

    file1 = text_file(file_root + '/train_images.txt', save_old)
    file2 = text_file(file_root + '/train_labels.txt', save_old)
    file3 = text_file(file_root + '/val_images.txt', save_old)
    file4 = text_file(file_root + '/val_labels.txt', save_old)

    data_length = len(all_image_paths)
    try:
        assert data_length == len(all_gt_paths)
    except Exception as e:
        print("Number of images: ", data_length)
        print("Number of labels: ", len(all_gt_paths))
        raise e

    train_length = int(ratio * data_length)

    train_image_paths, train_gt_paths = zip(*random.sample(list(zip(all_image_paths, all_gt_paths)), train_length))

    val_image_paths = [p for p in all_image_paths if p not in train_image_paths]
    val_gt_paths = [p for p in all_gt_paths if p not in train_gt_paths]

    for f in train_image_paths:
        file1.write(f)
        file1.write('\n')

    for f in train_gt_paths:
        file2.write(f)
        file2.write('\n')

    for f in val_image_paths:
        file3.write(f)
        file3.write('\n')

    for f in val_gt_paths:
        file4.write(f)
        file4.write('\n')

    file1.close()
    file2.close()
    file3.close()
    file4.close()


def get_image(path, mode=1):
    img = cv2.imread(path, mode)
    return img

