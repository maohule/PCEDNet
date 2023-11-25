import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()

cfg = __C
__C.DATA = edict()
__C.TRAIN = edict()
__C.VAL = edict()
__C.VIS = edict()

#------------------------------DATA------------------------
__C.DATA.STD_SIZE = (768,1024)
#__C.DATA.DATA_PATH = '/home/wanglin/workspace/pytorch_DL/PCCNet_vgg_2/CC/' +str(__C.DATA.STD_SIZE[0]) + 'x' + str(__C.DATA.STD_SIZE[1]) + '/shanghaitech_part_B' 
__C.DATA.DATA_PATH = 'path_to_the_datasets'          
__C.DATA.MEAN_STD = 'dataset_MEAN_STD' 
__C.DATA.DEN_ENLARGE = 1.

#------------------------------TRAIN------------------------
__C.TRAIN.INPUT_SIZE = 'input_size'
__C.TRAIN.SEED = 640
__C.TRAIN.RESUME = ''#model path
__C.TRAIN.BATCH_SIZE = 6 #imgs
__C.TRAIN.BCE_WIEGHT = 'BCE _WIEGHT_value'

__C.TRAIN.SEG_LR = 'SEG_LR_value'
__C.TRAIN.SEG_WIEGHT = 'SEG _WIEGHT_value'

__C.TRAIN.NUM_BOX = 20 #boxes

__C.TRAIN.GPU_ID = 'GPU_Device_number'

# base lr
__C.TRAIN.LR = 'TRAIN_LR'
__C.TRAIN.LR_DECAY = 0.995
__C.TRAIN.NUM_EPOCH_LR_DECAY = 1 # epoches

__C.TRAIN.MAX_EPOCH = 1000

# output 
__C.TRAIN.PRINT_FREQ = 10

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.TRAIN.EXP_NAME = 'PCC_Net' + now

__C.TRAIN.EXP_PATH = './exp'

#------------------------------VAL------------------------
__C.VAL.BATCH_SIZE = 1 # imgs
__C.VAL.FREQ = 1

#------------------------------VIS------------------------
__C.VIS.VISIBLE_NUM_IMGS = 20

#------------------------------MISC------------------------


#================================================================================
#================================================================================
#================================================================================  
