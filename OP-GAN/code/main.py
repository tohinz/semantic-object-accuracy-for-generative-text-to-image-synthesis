from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import TextDataset
from trainer import condGANTrainer as trainer

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from shutil import copyfile
import pickle

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    parser.add_argument('--cfg', dest='cfg_file', help='config file', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0,1,2,3')
    parser.add_argument('--resume', dest='resume', type=str, default='')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--net_g', dest='net_g', type=str, default='')
    parser.add_argument('--max_objects', type=int, default=10)
    args = parser.parse_args()
    return args


def get_dataset_indices(split="train", num_max_objects=10):
    if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
        label_path = os.path.join(os.path.join(cfg.DATA_DIR, split), 'labels_large.pickle')
    with open(label_path, "rb") as f:
        labels = pickle.load(f, encoding='latin1')
        labels = np.array(labels)
    dataset_indices = []

    for _i in range(num_max_objects+1):
        dataset_indices.append([])

    for index, label in enumerate(labels):
        for idx, l in enumerate(label):
            if l == -1:
                dataset_indices[idx].append(index)
                break
        else:
            dataset_indices[-1].append(index)

    return dataset_indices


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.net_g != "":
        cfg.TRAIN.NET_G = args.net_g
    print('Using config:')
    pprint.pprint(cfg)


    args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    print("Using seed {}".format(args.manualSeed))

    if args.resume == "":
        resume = False
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = '../output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    else:
        assert os.path.isdir(args.resume)
        resume = True
        output_dir = args.resume

    split_dir, bshuffle = 'train', True
    eval = False
    img_dir = "train/train2014"
    if not cfg.TRAIN.FLAG:
        split_dir = 'test'
        img_dir = "test/val2014"
        eval = True

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
            transforms.Resize((268, 268)),
            transforms.ToTensor()])

    if cfg.TRAIN.OPTIMIZE_DATA_LOADING:
        num_max_objects = 10
        dataset_indices = get_dataset_indices(num_max_objects=num_max_objects, split="train"\
                                              if cfg.TRAIN.FLAG else "test")
        dataset = TextDataset(cfg.DATA_DIR, img_dir, split_dir, base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform, eval=eval, use_generated_bboxes=cfg.TRAIN.GENERATED_BBOXES)
        assert dataset
        dataset_subsets = []
        dataloaders = []
        for max_objects in range(num_max_objects+1):
            subset = torch.utils.data.Subset(dataset, dataset_indices[max_objects])
            dataset_subsets.append(subset)
            dataloader = torch.utils.data.DataLoader(subset, batch_size=cfg.TRAIN.BATCH_SIZE[max_objects],
                                                     drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
            dataloaders.append(dataloader)

        algo = trainer(output_dir, dataloaders, dataset.n_words, dataset.ixtoword, resume)

    else:
        dataset = TextDataset(cfg.DATA_DIR, img_dir, split_dir, base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform, eval=eval, use_generated_bboxes=cfg.TRAIN.GENERATED_BBOXES)
        assert dataset
        dataloader = torch.utils.data.DataLoader( dataset, batch_size=cfg.TRAIN.BATCH_SIZE[0],
                                                  drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

        algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword, resume)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        if not resume:
            copyfile(sys.argv[0], output_dir + "/" + sys.argv[0])
            copyfile("trainer.py", output_dir + "/" + "trainer.py")
            copyfile("model.py", output_dir + "/" + "model.py")
            copyfile("miscc/utils.py", output_dir + "/" + "utils.py")
            copyfile("miscc/losses.py", output_dir + "/" + "losses.py")
            copyfile("GlobalAttention.py", output_dir + "/" + "GlobalAttention.py")
            copyfile("datasets.py", output_dir + "/" + "datasets.py")
            copyfile(args.cfg_file, output_dir + "/" + "cfg_file_train.yml")
        algo.train()
        end_t = time.time()
        print('Total time for training:', end_t - start_t)
    else:
        '''generate images from pre-extracted embeddings'''
        assert not cfg.TRAIN.OPTIMIZE_DATA_LOADING, "\"cfg.TRAIN.OPTIMIZE_DATA_LOADING\" not valid for sampling since we use"\
                                                    "generated bounding boxes at test time."
        use_generated_bboxes = cfg.TRAIN.GENERATED_BBOXES
        algo.sampling(split_dir, num_samples=500)  # generate images
