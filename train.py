from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from models.CC import CrowdCounter
from config import cfg
from loading_data import loading_data
from misc.utils import *
from misc.timer import Timer
import pdb
import cv2

exp_name = cfg.TRAIN.EXP_NAME
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)
log_txt_o = cfg.TRAIN.EXP_PATH + '/' + exp_name + '/' + exp_name + '.txt'


if not os.path.exists(cfg.TRAIN.EXP_PATH):
    os.mkdir(cfg.TRAIN.EXP_PATH)
    
pil_to_tensor = standard_transforms.ToTensor()

train_record = {'best_mae': 1e20, 'mse':1e20,'corr_loss': 0, 'corr_epoch': -1, 'best_model_name': ''}

train_set, train_loader, val_set, val_loader, restore_transform = loading_data()

_t = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

rand_seed = cfg.TRAIN.SEED    
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def log(f,txt):
    f.write(txt+'\n')

def main():

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt_o, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = CrowdCounter(ce_weights=train_set.wts,modelname='res_backbone')

    net.train()

    optimizer = optim.Adam([
                            {'params': [param for name, param in net.named_parameters() if 'seg' in name], 'lr': cfg.TRAIN.SEG_LR},
                            {'params': [param for name, param in net.named_parameters() if 'base' in name], 'lr': cfg.TRAIN.SEG_LR},
                            {'params': [param for name, param in net.named_parameters() if 'seg' not in name and 'base' not in name], 'lr': cfg.TRAIN.LR}
                          ])
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    
    f=open('./train_log.log','w')   

    i_tb = 0
    for epoch in range(cfg.TRAIN.MAX_EPOCH):       

        _t['train time'].tic()
        i_tb,model_path = train(train_loader, net, optimizer, epoch, i_tb,f)
        _t['train time'].toc(average=False)
        print( 'train time of one epoch: {:.2f}s'.format(_t['train time'].diff) )
        log_txt='train time of one epoch: {:.2f}s'.format(_t['train time'].diff)
        log(f,log_txt)
        if epoch%cfg.VAL.FREQ!=0:
            continue
        _t['val time'].tic()
        validate(val_loader, model_path, epoch, restore_transform,f)
        _t['val time'].toc(average=False)
        print( 'val time of one epoch: {:.2f}s'.format(_t['val time'].diff))
        log_txt1='val time of one epoch: {:.2f}s'.format(_t['val time'].diff)
        log(f,log_txt1)
        scheduler.step()
    log(f,'Done')
    f.close()

def train(train_loader, net, optimizer, epoch, i_tb,f):
    
    for i, data in enumerate(train_loader, 0):
        _t['iter time'].tic()
        
        img, gt_map, gt_cnt,gt_seg,gt_seg_sec,gt_fore,gt_fore_sec = data

        img = Variable(img).cuda()
        gt_map = Variable(gt_map).cuda()
        gt_seg = Variable(gt_seg).cuda()
        gt_seg_sec = Variable(gt_seg_sec).cuda()
        gt_fore = Variable(gt_fore).cuda()
        gt_fore_sec = Variable(gt_fore_sec).cuda()

        optimizer.zero_grad()
        pred_map, pred_seg, pred_seg_sec,pred_fore,pred_fore_sec = net(img, gt_map, gt_seg,gt_seg_sec,gt_fore,gt_fore_sec,train_mode=True)

        loss = net.loss
        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        if (i + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            
            loss1,loss3,loss4,loss5,loss6 = net.f_loss()

            i_tb = i_tb + 1
            writer.add_scalar('train_loss_mse', loss1.item(), i_tb)
            writer.add_scalar('train_loss_seg', loss3.item(), i_tb)
            writer.add_scalar('train_loss_seg_sec', loss4.item(), i_tb)
            writer.add_scalar('train_loss_fore', loss5.item(), i_tb)
            writer.add_scalar('train_loss_fore_sec', loss6.item(), i_tb)
            writer.add_scalar('train_loss', loss.item(), i_tb)

            _t['iter time'].toc(average=False)
            print( '[ep %d][it %d][loss %.8f %.8f %.4f %.4f %.4f %.4f][%.2fs]' % \
                    (epoch + 1, i + 1, loss.item(), loss1.item(), loss3.item(),loss4.item(),loss5.item(),loss6.item(), _t['iter time'].diff) )
            log_txt2='[ep %d][it %d][loss %.8f %.8f %.4f %.4f %.4f %.4f][%.2fs]' % \
                    (epoch + 1, i + 1, loss.item(), loss1.item(), loss3.item(),loss4.item(),loss5.item(),loss6.item(), _t['iter time'].diff)
            log(f,log_txt2)

            # pdb.set_trace()
            print( '        [cnt: gt: %.1f pred: %.6f]' % (gt_cnt[0]/cfg.DATA.DEN_ENLARGE, pred_map[0,:,:,:].sum().item()/cfg.DATA.DEN_ENLARGE) )
            log_txt3='        [cnt: gt: %.1f pred: %.6f]' % (gt_cnt[0]/cfg.DATA.DEN_ENLARGE, pred_map[0,:,:,:].sum().item()/cfg.DATA.DEN_ENLARGE) 
            log(f,log_txt3)              
    
    snapshot_name = 'all_ep_%d' % (epoch + 1)
    # save model
    to_saved_weight = []

    if len(cfg.TRAIN.GPU_ID)>1:
        to_saved_weight = net.module.state_dict()                
    else:
        to_saved_weight = net.state_dict()
    model_path = os.path.join(cfg.TRAIN.EXP_PATH, exp_name, snapshot_name + '.pth')
    torch.save(to_saved_weight, model_path)

    return i_tb,model_path

def validate(val_loader, model_path, epoch, restore,f):
    net = CrowdCounter(ce_weights=train_set.wts)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    log_txt4='='*50
    log(f,log_txt4)

    mae = 0.0
    mse = 0.0

    for vi, data in enumerate(val_loader, 0):
        img, gt_map, gt_cnt, gt_seg,gt_seg_sec,gt_fore,gt_fore_sec = data
        # pdb.set_trace()
        with torch.no_grad():
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()
            gt_seg = Variable(gt_seg).cuda()
            gt_seg_sec = Variable(gt_seg_sec).cuda()
            gt_fore = Variable(gt_fore).cuda()
            gt_fore_sec = Variable(gt_fore_sec).cuda()

            pred_map,pred_seg,pred_seg_sec,pred_fore,pred_fore_sec = net(img, gt_map, gt_seg, gt_seg_sec,gt_fore,gt_fore_sec,train_mode=True)

            pred_map = pred_map.data.cpu().numpy()/cfg.DATA.DEN_ENLARGE
            gt_map = gt_map.data.cpu().numpy()/cfg.DATA.DEN_ENLARGE

            gt_count = np.sum(gt_map)
            pred_cnt = np.sum(pred_map)

            gt_map=gt_map.squeeze(0)
            gt_map = cv2.resize(gt_map,(pred_map.shape[3],pred_map.shape[2]),interpolation = cv2.INTER_CUBIC)
            gt_map=gt_map[np.newaxis,:,:]

            pred_seg = pred_seg.cpu().max(1)[1].squeeze_(1).data.numpy()
            gt_seg_sec = gt_seg_sec.data.cpu().numpy()
            pred_seg_sec = pred_seg_sec.cpu().max(1)[1].squeeze_(1).data.numpy()

            pred_fore = pred_fore.cpu().max(1)[1].squeeze_(1).data.numpy()
            gt_fore_sec = gt_fore_sec.data.cpu().numpy()
            pred_fore_sec = pred_fore_sec.cpu().max(1)[1].squeeze_(1).data.numpy()

            pred_seg=pred_seg.squeeze(0)
            pred_seg = Image.fromarray(pred_seg.astype(np.uint8))
            pred_seg =pred_seg.resize((gt_map.shape[2],gt_map.shape[1]), Image.NEAREST)

            pred_seg=np.array(pred_seg)
            pred_seg=pred_seg[np.newaxis,:,:]

            pred_seg_sec=pred_seg_sec.squeeze(0)
            pred_seg_sec = Image.fromarray(pred_seg_sec.astype(np.uint8))
            pred_seg_sec =pred_seg_sec.resize((gt_map.shape[2],gt_map.shape[1]), Image.NEAREST)

            pred_seg_sec=np.array(pred_seg_sec)
            pred_seg_sec=pred_seg_sec[np.newaxis,:,:]

            pred_fore=pred_fore.squeeze(0)
            pred_fore = Image.fromarray(pred_fore.astype(np.uint8))
            pred_fore =pred_fore.resize((gt_map.shape[2],gt_map.shape[1]), Image.NEAREST)

            pred_fore=np.array(pred_fore)
            pred_fore=pred_fore[np.newaxis,:,:]

            pred_fore_sec=pred_fore_sec.squeeze(0)
            pred_fore_sec = Image.fromarray(pred_fore_sec.astype(np.uint8))
            pred_fore_sec =pred_fore_sec.resize((gt_map.shape[2],gt_map.shape[1]), Image.NEAREST)

            pred_fore_sec=np.array(pred_fore_sec)
            pred_fore_sec=pred_fore_sec[np.newaxis,:,:]

            gt_seg_sec=gt_seg_sec.squeeze(0)
            gt_seg_sec = Image.fromarray(gt_seg_sec.astype(np.uint8))
            gt_seg_sec =gt_seg_sec.resize((gt_map.shape[2],gt_map.shape[1]), Image.NEAREST)

            gt_seg_sec=np.array(gt_seg_sec)
            gt_seg_sec=gt_seg_sec[np.newaxis,:,:]

            gt_fore_sec=gt_fore_sec.squeeze(0)
            gt_fore_sec = Image.fromarray(gt_fore_sec.astype(np.uint8))
            gt_fore_sec =gt_fore_sec.resize((gt_map.shape[2],gt_map.shape[1]), Image.NEAREST)

            gt_fore_sec=np.array(gt_fore_sec)
            gt_fore_sec=gt_fore_sec[np.newaxis,:,:]

            mae += abs(gt_count-pred_cnt)
            mse += ((gt_count-pred_cnt)*(gt_count-pred_cnt))

            x = []
            if vi==0:
                for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, pred_seg, gt_seg_sec,pred_seg_sec,pred_fore,pred_fore_sec,gt_fore_sec)):
                    if idx>cfg.VIS.VISIBLE_NUM_IMGS:
                        break
            #         # pdb.set_trace()
                    pil_input = restore(tensor[0])
                    #pil_input=pil_input.resize((pil_input.size[0]//4,pil_input.size[1]//4))
                    pil_input=pil_input.resize((gt_map.shape[2],gt_map.shape[1]))
                    pil_label = torch.from_numpy(tensor[2]/(tensor[2].max()+1e-10)).repeat(3,1,1)
                    pil_output = torch.from_numpy(tensor[1]/(tensor[1].max()+1e-10)).repeat(3,1,1)
                    
                    pil_gt_seg = torch.from_numpy(tensor[4]).repeat(3,1,1).float()
                    pil_pred_seg = torch.from_numpy(tensor[3]).repeat(3,1,1).float()
                    pil_pred_seg_sec = torch.from_numpy(tensor[5]).repeat(3,1,1).float()
                    pil_gt_fore = torch.from_numpy(tensor[8]).repeat(3,1,1).float()
                    pil_pred_fore = torch.from_numpy(tensor[6]).repeat(3,1,1).float()
                    pil_pred_fore_sec = torch.from_numpy(tensor[7]).repeat(3,1,1).float()
            #         # pdb.set_trace()
                    
                    x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output, pil_gt_seg, pil_pred_seg, pil_pred_seg_sec,pil_gt_fore,pil_pred_fore,pil_pred_fore_sec])
                x = torch.stack(x, 0)
                x = vutils.make_grid(x, nrow=9, padding=9)
                writer.add_image(exp_name + '_epoch_' + str(epoch+1), (x.numpy()*255).astype(np.uint8))

    mae = mae/val_set.get_num_samples()
    mse = np.sqrt(mse/val_set.get_num_samples())

    if mae < train_record['best_mae']:
        train_record['best_mae'] = mae
        train_record['mse'] = mse
        train_record['corr_epoch'] = epoch + 1
    print('='*50)
    print( exp_name )
    # pdb.set_trace()
    print( '[best] [mae %.1f mse %.1f], [epoch %d]' % (train_record['best_mae'], train_record['mse'], train_record['corr_epoch']) )
    log_txt10='[best] [mae %.1f mse %.1f], [loss %.8f], [epoch %d]' % (train_record['best_mae'], train_record['mse'], train_record['corr_loss'], train_record['corr_epoch'])
    print('='*50)
    log(f,log_txt10)
    log_txt11='='*50
    log(f,log_txt11)

if __name__ == '__main__':
    main()








