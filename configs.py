import numpy as np

# DATASET PARAMETERS
TRAIN_DIR = 'D:/VOCdata/VOC2012AUG/'
VAL_DIR = TRAIN_DIR
stages = 3
TRAIN_LIST = ['D:/VOCdata/VOC2012AUG/train_aug.txt'] * stages
VAL_LIST = ['D:/VOCdata/VOC2012AUG/val.txt'] * stages
SHORTER_SIDE = [350] * stages
CROP_SIZE = [500] * stages
NORMALISE_PARAMS = [1. / 255,  # SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))]  # STD
BATCH_SIZE = [4] * stages
NUM_WORKERS = 16
NUM_CLASSES = [21] * stages
LOW_SCALE = [0.5] * stages
HIGH_SCALE = [2.0] * stages
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = '50'
ENC_PRETRAINED = False  # pre-trained on ImageNet or randomly initialised

# GENERAL
EVALUATE = False
FREEZE_BN = [False] * stages
NUM_SEGM_EPOCHS = [100] * stages
PRINT_EVERY = 10
RANDOM_SEED = 42
SNAPSHOT_DIR = './ckpt/'
CKPT_PATH = './ckpt/checkpoint.pth.tar'
MODEL_PATH = './models/my_model.pth'
VAL_EVERY = [5] * stages  # how often to record validation scores

# OPTIMISERS' PARAMETERS
LR_ENC = [5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
LR_DEC = [5e-3, 2.5e-3, 1e-3]
MOM_ENC = [0.9] * stages  # TO FREEZE, PUT 0
MOM_DEC = [0.9] * stages
WD_ENC = [1e-5] * stages  # TO FREEZE, PUT 0
WD_DEC = [1e-5] * stages
OPTIM_DEC = 'sgd'

palette = [(0, 0, 0),
           (128, 0, 0),
           (0, 128, 0),
           (128, 128, 0),
           (0, 0, 128),
           (128, 0, 128),
           (0, 128, 128),
           (128, 128, 128),
           (64, 0, 0),
           (192, 0, 0),
           (64, 128, 0),
           (192, 128, 0),
           (64, 0, 128),
           (192, 0, 128),
           (64, 128, 128),
           (192, 128, 128),
           (0, 64, 0),
           (128, 64, 0),
           (0, 192, 0),
           (128, 192, 0),
           (0, 64, 128)]

classes = {'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4,
             'bottle'    : 5,  'bus'       : 6,  'car'         : 7,  'cat'          : 8,
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12,
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted-plant' : 16,
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv/monitor'   : 20}
