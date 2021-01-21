from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from easydict import EasyDict as edict

logger = logging.getLogger()

__C = edict()
cfg = __C

__C.DATASET_NAME = 'coco'
__C.CONFIG_NAME = ''
__C.DATA_DIR = 'data'
__C.OUTPUT_DIR = 'output'
__C.CUDA = True
__C.WORKERS = 6
__C.SEED = -1
__C.DEBUG = False
__C.DEBUG_NUM_DATAPOINTS = 100

__C.RNN_TYPE = 'LSTM'   # 'GRU'

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 64

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = [24]
__C.TRAIN.MAX_EPOCH = 120
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.RNN_GRAD_CLIP = 0.25
__C.TRAIN.FLAG = True
__C.TRAIN.NET_E = ''
__C.TRAIN.NET_G = ''
__C.TRAIN.BBOX_LOSS = True
__C.TRAIN.B_NET_D = True
__C.TRAIN.OPTIMIZE_DATA_LOADING = True
__C.TRAIN.EMPTY_CACHE = False
__C.TRAIN.GENERATED_BBOXES = False

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0

# Modal options
__C.GAN = edict()
__C.GAN.DISC_FEAT_DIM = 96
__C.GAN.GEN_FEAT_DIM = 48
__C.GAN.GLOBAL_Z_DIM = 100
__C.GAN.LOCAL_Z_DIM = 32
__C.GAN.TEXT_CONDITION_DIM = 100
__C.GAN.INIT_LABEL_DIM = 100
__C.GAN.NEXT_LABEL_DIM = 256 // 2
__C.GAN.RESIDUAL_NUM = 3
__C.GAN.B_ATTENTION = True
__C.GAN.B_DCGAN = False
__C.GAN.LAYOUT_SPATIAL_DIM = 16

__C.TEXT = edict()
__C.TEXT.CAPTIONS_PER_IMAGE = 10
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.WORDS_NUM = 12
__C.TEXT.CLASSES_NUM = 81


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                logger.info('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)
