from matplotlib import pyplot as plt
import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.resnet_big import ori
from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image
import cv2

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

exp_name = './DULR-display-save-mat'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

mean_std = cfg.DATA.MEAN_STD
img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

wts = torch.FloatTensor(
        [ 0.07259259,  0.05777778,  0.10148148,  0.10592593,  0.10925926,\
        0.11      ,  0.11037037,  0.11074074,  0.11111111,  0.11074074]    
            )
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()

dataRoot = 'PATH_TO_DATASET'

model_path ='PATH_TO_MODEL'

os.makedirs('DULR-display-save-mat/pred', exist_ok=True)
os.makedirs('DULR-display-save-mat/dens', exist_ok=True)
os.makedirs('DULR-display-save-mat/seg', exist_ok=True)
os.makedirs('DULR-display-save-mat/fore', exist_ok=True)
os.makedirs('DULR-display-save-mat/seg_sec', exist_ok=True)
os.makedirs('DULR-display-save-mat/fore_sec', exist_ok=True)


def main():
    # file_list = [filename for filename in os.listdir(dataRoot+'/img/') if os.path.isfile(os.path.join(dataRoot+'/img/',filename))]
    file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]
    # pdb.set_trace()
    test(file_list[0], model_path)

def test(file_list, model_path):

    net = CrowdCounter(ce_weights=wts)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0

    for filename in file_list:
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]

        denname = dataRoot + '/den/' + filename_no_ext + '.csv'

        den = pd.read_csv(denname, sep=',',header=None).values
        den = den.astype(np.float32, copy=False)
        #print('den_type:',type(den))
        img = Image.open(imgname)

        # prepare
        
        wd_1, ht_1 = img.size
        '''
        if wd_1 < cfg.DATA.STD_SIZE[1]:
            dif = cfg.DATA.STD_SIZE[1] - wd_1
            pad = np.zeros([ht_1,dif])
            img = np.array(img)
            den = np.array(den)
            img = np.hstack((img,pad))
            img = Image.fromarray(img.astype(np.uint8))
            den = np.hstack((den,pad))
            
        if ht_1 < cfg.DATA.STD_SIZE[0]:
            dif = cfg.DATA.STD_SIZE[0] - ht_1
            pad = np.zeros([dif,wd_1])
            img = np.array(img)
            den = np.array(den)
            # pdb.set_trace()
            img = np.vstack((img,pad))
            img = Image.fromarray(img.astype(np.uint8))

            den = np.vstack((den,pad))
        '''
        img = img_transform(img)



        gt = np.sum(den)
        # den = Image.fromarray(den)

        #img = img*255.

        img = Variable(img[None,:,:,:],volatile=True).cuda()

        #forward
        with torch.no_grad():
            pred_map,pred_seg,pred_seg_sec,pred_fore,pred_fore_sec = net.test_forward(img)

        pred_map = pred_map.cpu().data.numpy()[0,0,:,:]

        pred = np.sum(pred_map)

        den_down = cv2.resize(den,(den.shape[1]//2,den.shape[0]//2),interpolation = cv2.INTER_CUBIC)*4
        gt_down=np.sum(den_down)

        den = torch.from_numpy(np.array(den))*cfg.DATA.DEN_ENLARGE
        #den_map = den.cpu().data.numpy()[0,0,:,:]
        den_map = den_down

    
        mae += abs(gt-pred)
        mse += ((gt-pred)*(gt-pred))

        #print('pred_seg.shape:',pred_seg.shape)
        pred_seg = pred_seg.cpu().max(1)[1].squeeze_(1).data.numpy()
        pred_seg=pred_seg.squeeze(0)
        
        pred_seg_sec = pred_seg_sec.cpu().max(1)[1].squeeze_(1).data.numpy()
        pred_seg_sec=pred_seg_sec.squeeze(0)

        pred_fore = pred_fore.cpu().max(1)[1].squeeze_(1).data.numpy()
        pred_fore=pred_fore.squeeze(0)

        pred_fore_sec = pred_fore_sec.cpu().max(1)[1].squeeze_(1).data.numpy()
        pred_fore_sec=pred_fore_sec.squeeze(0)

        plt.imsave(os.path.join('DULR-display-save-mat/pred', f'{filename_no_ext}_{pred:.2f}_{gt:.2f}.png'), pred_map, cmap='jet')
        plt.imsave(os.path.join('DULR-display-save-mat/dens', f'[{filename_no_ext}].png'), den, cmap='jet')
        plt.imsave(os.path.join('DULR-display-save-mat/seg', f'[{filename_no_ext}].png'),pred_seg,cmap='Greys')
        plt.imsave(os.path.join('DULR-display-save-mat/seg_sec', f'[{filename_no_ext}].png'), pred_seg_sec, cmap='Greys')
        plt.imsave(os.path.join('DULR-display-save-mat/fore', f'[{filename_no_ext}].png'),pred_fore , cmap='Greys')
        plt.imsave(os.path.join('DULR-display-save-mat/fore_sec', f'[{filename_no_ext}].png'), pred_fore_sec, cmap='Greys')

    mae = mae/len(file_list)
    mse = np.sqrt(mse/len(file_list))
    print('mae:',mae)
    print('mse:',mse)



if __name__ == '__main__':
    main()




