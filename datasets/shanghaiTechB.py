import numpy as np
import os
import random
import pandas as pd
import sys
import torch
from torch.utils import data
from PIL import Image
from config import cfg
import cv2
import pdb
#import h5py

# train info
min_gt_count = 2.7126654189217972e-05
max_gt_count = 0.001306603490202515

wts = torch.FloatTensor(
       [ 0.10194444,  0.07416667,  0.08361111,  0.09277778,  0.10388889,\
        0.10416667,  0.10805556,  0.11      ,  0.11111111,  0.11027778]    
            )
box_num = cfg.TRAIN.NUM_BOX

class SHT_B(data.Dataset):
    def __init__(self, data_path, main_transform=None, img_transform=None, gt_transform=None,train_mode=True):
        #pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        #num_classes: total number of classes into which the crowd count is divided (default: 10 as used in the paper)
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.seg_path = data_path + '/seg'
        #self.fore_path = data_path + '/seg_2'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.wts = wts

    def __getitem__(self, index):
        fname = self.data_files[index]

        img, den, seg = self.read_image_and_gt(fname)
      
        if self.main_transform is not None:
            img, den, seg = self.main_transform(img,den,seg)
        if self.img_transform is not None:
            img = self.img_transform(img)

        img = img# *255.

        den = np.array(den)

        fore_0=1-np.array(seg).astype(np.uint8)
        fore_1=fore_0>0
        fore=Image.fromarray(fore_1)
        seg_one=seg.resize((den.shape[1]//8,den.shape[0]//8), Image.NEAREST)
        seg_sec=seg.resize((den.shape[1]//4,den.shape[0]//4), Image.NEAREST)
        fore_one=fore.resize((den.shape[1]//8,den.shape[0]//8), Image.NEAREST)
        fore_sec=fore.resize((den.shape[1]//4,den.shape[0]//4), Image.NEAREST)

        den = cv2.resize(den,(den.shape[1]//2,den.shape[0]//2),interpolation = cv2.INTER_CUBIC)*4
        den=Image.fromarray(den)

        den = torch.from_numpy(np.array(den))*cfg.DATA.DEN_ENLARGE
        seg_one = torch.from_numpy(np.array(seg_one).astype(np.uint8)).long()
        seg_sec = torch.from_numpy(np.array(seg_sec).astype(np.uint8)).long()
        fore_one = torch.from_numpy(np.array(fore_one).astype(np.uint8)).long()
        fore_sec = torch.from_numpy(np.array(fore_sec).astype(np.uint8)).long()
        gt_count = den.sum()    

        return img, den, gt_count, seg_one,seg_sec,fore_one,fore_sec

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        img = Image.open(os.path.join(self.img_path,fname))
        seg = Image.open(os.path.join(self.seg_path,fname.split('.')[0]+'.png'))
        if img.mode == 'L':
            img = img.convert('RGB')
        wd_1, ht_1 = img.size

        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        den = den.astype(np.float32, copy=False)
        
        gt_count = torch.from_numpy(den).sum()

        # add padding

        if wd_1 < cfg.DATA.STD_SIZE[1]:
            dif = cfg.DATA.STD_SIZE[1] - wd_1
            pad = np.zeros([ht_1,dif])
            img = np.hstack((np.array(img),pad))
            seg = np.hstack((np.array(seg),pad))
            den = np.hstack((np.array(den),pad))

            img = Image.fromarray(img.astype(np.uint8))
            seg = Image.fromarray(seg.astype(np.uint8))
            
        if ht_1 < cfg.DATA.STD_SIZE[0]:
            dif = cfg.DATA.STD_SIZE[0] - ht_1
            pad = np.zeros([dif,wd_1])
            img = np.vstack((np.array(img),pad))
            seg = np.vstack((np.array(seg),pad))
            den = np.vstack((np.array(den),pad))

            img = Image.fromarray(img.astype(np.uint8))
            seg = Image.fromarray(seg.astype(np.uint8))
            
        den = Image.fromarray(den)
        return img, den, seg

    def get_num_samples(self):
        return self.num_samples       
            
        
