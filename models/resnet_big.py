import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC, convDU, convLR
from torchvision import models
from misc.utils import weights_normal_init, initialize_weights

class ori(nn.Module):

    def __init__(self, bn=False, num_classes=10):
        super(ori, self).__init__()

        #vgg = models.vgg16(pretrained=True)
        res = models.resnet50(pretrained=True)
        self.base_layer = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu)
        self.base_layer1=nn.Sequential(
            res.maxpool,
            res.layer1)
        self.layer2=res.layer2
        self.base_own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)
        self.base_own_reslayer_3.load_state_dict(res.layer3.state_dict())

        self.squz_layer = Conv2d(1024, 256, 1, same_padding=True, NL='prelu', bn=bn)

        # generate dense map
        self.den_stage_1 = nn.Sequential(Conv2d( 256, 128, 7, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(128, 64, 5, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(64, 32, 5, same_padding=True, NL='prelu', bn=bn),
                                     Conv2d(32, 32, 5, same_padding=True, NL='prelu', bn=bn))

        self.den_stage_2 = nn.Sequential(Conv2d( 32, 32, 3, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d( 32, 16, 3, same_padding=True, NL='prelu', bn=bn,dilation=2),
                                        Conv2d( 16, 8, 3, same_padding=True, NL='prelu', bn=bn,dilation=2))

        self.den_stage_3 = nn.Sequential(Conv2d( 32, 32, 3, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d( 32, 16, 3, same_padding=True, NL='prelu', bn=bn,dilation=2),
                                        Conv2d( 16, 8, 3, same_padding=True, NL='prelu', bn=bn,dilation=2))

        # generate background map
        self.seg_stage = nn.Sequential(Conv2d( 32, 32, 1, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d( 32, 16, 3, same_padding=True, NL='prelu', bn=bn,dilation=2),
                                        Conv2d( 16, 8, 3, same_padding=True, NL='prelu', bn=bn,dilation=2))

        self.seg_stage_sec = nn.Sequential(Conv2d( 32, 32, 1, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d( 32, 16, 3, same_padding=True, NL='prelu', bn=bn,dilation=2),
                                        Conv2d( 16, 8, 3, same_padding=True, NL='prelu', bn=bn,dilation=2))
        # generate foreground map
        self.seg_fore_stage = nn.Sequential(Conv2d( 32, 32, 1, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d( 32, 16, 3, same_padding=True, NL='prelu', bn=bn,dilation=2),
                                        Conv2d( 16, 8, 3, same_padding=True, NL='prelu', bn=bn,dilation=2))

        self.seg_fore_stage_sec = nn.Sequential(Conv2d( 32, 32, 1, same_padding=True, NL='prelu', bn=bn),
                                        Conv2d( 32, 16, 3, same_padding=True, NL='prelu', bn=bn,dilation=2),
                                        Conv2d( 16, 8, 3, same_padding=True, NL='prelu', bn=bn,dilation=2))

        self.seg_pred = Conv2d(8, 2, 1, same_padding=True, NL='relu', bn=bn)
        self.seg_pred_sec = Conv2d(8, 2, 1, same_padding=True, NL='relu', bn=bn)
        self.seg_fore_pred = Conv2d(8, 2, 1, same_padding=True, NL='relu', bn=bn)
        self.seg_fore_pred_sec = Conv2d(8, 2, 1, same_padding=True, NL='relu', bn=bn)

        self.conv_32_3 = Conv2d(24, 32, 3, same_padding=True, NL='relu', bn=bn)
        self.conv_32_3_sec = Conv2d(24, 32, 3, same_padding=True, NL='relu', bn=bn)

        self.den_pred = Conv2d(32, 1, 1, same_padding=True, NL='relu', bn=bn)

        weights_normal_init(self.squz_layer, self.den_stage_1, self.den_stage_2, self.den_stage_3, self.conv_32_3, self.conv_32_3_sec, self.den_pred)
        initialize_weights(self.seg_stage,self.seg_stage_sec, self.seg_pred,self.seg_pred_sec,self.seg_fore_stage,self.seg_fore_stage_sec,self.seg_fore_pred,self.seg_fore_pred_sec)

        
    def forward(self, im_data):
        x_base1 = self.base_layer(im_data)
        x_base2 = self.base_layer1(x_base1)
        x_base3 = self.layer2(x_base2)
        x_base = self.base_own_reslayer_3(x_base3)
        x_base = self.squz_layer(x_base)
        x_map_1 = self.den_stage_1(x_base)

        x_map = self.den_stage_2(x_map_1)

        x_seg_fea = self.seg_stage(x_map_1)
        x_fore_fea=self.seg_fore_stage(x_map_1)
        x_seg = self.seg_pred(x_seg_fea)
        x_fore = self.seg_fore_pred(x_fore_fea)
        x_sub=x_map-x_seg_fea
        #x_sub_trans = self.trans_den(x_sub)
        x_fore_strong=x_map*x_fore_fea
        x_cat=torch.cat((x_sub,x_fore_fea,x_fore_strong),1)
        x_cat_32=self.conv_32_3(x_cat)
        x_sum=x_cat_32+x_map_1
        x_sum_up=F.interpolate(x_sum, size=x_base2.shape[2:], mode='bilinear', align_corners=False)

        x_map_sec = self.den_stage_3(x_sum_up)
        x_seg_fea_sec = self.seg_stage_sec(x_sum_up)
        x_fore_fea_sec=self.seg_fore_stage_sec(x_sum_up)
        x_seg_sec = self.seg_pred_sec(x_seg_fea_sec)
        x_fore_sec=self.seg_fore_pred_sec(x_fore_fea_sec)
        x_sub_sec=x_map_sec-x_seg_fea_sec
        x_fore_strong_sec=x_map_sec*x_fore_fea_sec
        x_cat_sec=torch.cat((x_sub_sec,x_fore_fea_sec,x_fore_strong_sec),1)
        x_cat_sec_32=self.conv_32_3_sec(x_cat_sec)
        x_sum_sec=x_cat_sec_32+x_sum_up
        x_sum_sec_up=F.interpolate(x_sum_sec, size=x_base1.shape[2:], mode='bilinear', align_corners=False)

        x_pred = self.den_pred(x_sum_sec_up)
        return x_pred, x_seg,x_seg_sec,x_fore,x_fore_sec

def make_res_layer(block, planes, blocks, stride=1):
    downsample = None
    inplanes = 512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            planes *
            self.expansion,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

