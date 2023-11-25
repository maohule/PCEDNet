import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .resnet_big import ori
from config import cfg
from misc.utils import CrossEntropyLoss2d
import cv2
from PIL import Image
import numpy as np
class CrowdCounter(nn.Module):
    def __init__(self, ce_weights=None, modelname='ori'):
        super(CrowdCounter, self).__init__()        


        from .resnet_big import ori

        self.CCN = ori()        

        self.CCN=self.CCN.cuda()

        if ce_weights is not None:
            ce_weights = torch.Tensor(ce_weights)
            ce_weights = ce_weights.cuda()
        self.loss_mse_fn = nn.MSELoss().cuda()
        self.loss_cel_fn = CrossEntropyLoss2d().cuda()

    @property
    def loss(self):
        return self.loss_mse  + cfg.TRAIN.SEG_WIEGHT*self.loss_seg + cfg.TRAIN.SEG_WIEGHT*self.loss_seg_sec+cfg.TRAIN.SEG_WIEGHT*self.loss_fore + cfg.TRAIN.SEG_WIEGHT*self.loss_fore_sec

    def f_loss(self):
        return self.loss_mse,self.loss_seg,self.loss_seg_sec,self.loss_fore,self.loss_fore_sec
    
    def forward(self, img, gt_map, gt_seg,gt_seg_sec,gt_fore,gt_fore_sec,train_mode=True):                               
        density_map, pred_seg,pred_seg_sec,pred_fore,pred_fore_sec = self.CCN(img)

        if train_mode==True:
            self.loss_mse, self.loss_seg, self.loss_seg_sec,self.loss_fore,self.loss_fore_sec= self.build_loss(density_map,  pred_seg, pred_seg_sec,gt_map, gt_seg,gt_seg_sec,pred_fore,pred_fore_sec,gt_fore,gt_fore_sec)
        else:
            gt_map=gt_map.squeeze(0).data.cpu().numpy()
            gt_map = cv2.resize(gt_map,(density_map.data.cpu().numpy().shape[3],density_map.data.cpu().numpy().shape[2]),interpolation = cv2.INTER_CUBIC)
            gt_map=torch.tensor(gt_map).unsqueeze(0).cuda()
            
            gt_seg=Image.fromarray(gt_seg.data.cpu().numpy().squeeze(0).astype(np.uint8))
            gt_seg=gt_seg.resize((pred_seg.shape[3],pred_seg.shape[2]), Image.NEAREST)
            gt_seg = torch.from_numpy(np.array(gt_seg).astype(np.uint8)).long().unsqueeze(0).cuda()
            gt_seg_sec=Image.fromarray(gt_seg_sec.data.cpu().numpy().squeeze(0).astype(np.uint8))
            gt_seg_sec=gt_seg_sec.resize((pred_seg_sec.shape[3],pred_seg_sec.shape[2]), Image.NEAREST)
            gt_seg_sec = torch.from_numpy(np.array(gt_seg_sec).astype(np.uint8)).long().unsqueeze(0).cuda()

            gt_fore=Image.fromarray(gt_fore.data.cpu().numpy().squeeze(0).astype(np.uint8))
            gt_fore=gt_fore.resize((pred_seg.shape[3],pred_seg.shape[2]), Image.NEAREST)
            gt_fore = torch.from_numpy(np.array(gt_fore).astype(np.uint8)).long().unsqueeze(0).cuda()
            gt_fore_sec=Image.fromarray(gt_fore_sec.data.cpu().numpy().squeeze(0).astype(np.uint8))
            gt_fore_sec=gt_fore_sec.resize((pred_seg_sec.shape[3],pred_seg_sec.shape[2]), Image.NEAREST)
            gt_fore_sec = torch.from_numpy(np.array(gt_fore_sec).astype(np.uint8)).long().unsqueeze(0).cuda()
            self.loss_mse, self.loss_seg, self.loss_seg_sec,self.loss_fore,self.loss_fore_sec= self.build_loss(density_map,  pred_seg, pred_seg_sec,gt_map, gt_seg,gt_seg_sec,pred_fore,pred_fore_sec,gt_fore,gt_fore_sec)

        return density_map, pred_seg,pred_seg_sec,pred_fore,pred_fore_sec
    
    def build_loss(self, density_map, pred_seg, pred_seg_sec,gt_data, gt_seg,gt_seg_sec,pred_fore,pred_fore_sec,gt_fore,gt_fore_sec):
        loss_mse = self.loss_mse_fn(density_map.squeeze(), gt_data.squeeze())  
        # pdb.set_trace()
        loss_seg = self.loss_cel_fn(pred_seg, gt_seg)
        loss_seg_sec = self.loss_cel_fn(pred_seg_sec, gt_seg_sec)  
        loss_fore = self.loss_cel_fn(pred_fore, gt_fore)
        loss_fore_sec = self.loss_cel_fn(pred_fore_sec, gt_fore_sec)    
        return loss_mse, loss_seg,loss_seg_sec,loss_fore,loss_fore_sec

    def test_forward(self, img):                               
        density_map,pred_seg, pred_seg_sec,pred_fore,pred_fore_sec= self.CCN(img)            
            
        return density_map, pred_seg,pred_seg_sec,pred_fore,pred_fore_sec        

