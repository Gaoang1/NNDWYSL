from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F
import math
from torchvision.ops import DeformConv2d

class ECAG(nn.Module):
    def __init__(self, in_channels):
        super(ECAG, self).__init__()
        b = 1
        gamma = 2
        kernel_size = int(abs((math.log(in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.reduce_channels = nn.Conv2d(in_channels * 2, in_channels, 1)
        self.conv_1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.maxpool(x)
        y2 = self.avgpool(x)

        y = torch.cat([y1, y2], dim=1)
        y = self.reduce_channels(y)

        y = self.conv_1d(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        out = self.sigmoid(y)
        return out

class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2, g * 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        return self.dcn(x, offset, mask)

class FSM(nn.Module):
    def __init__(self, in_channels):
        super(FSM, self).__init__()
        self.conv_atten = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn_atten = nn.GroupNorm(num_groups=64, num_channels=256)
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_uniform_(self.conv_atten.weight, a=1)

    def forward(self, x):
        atten = self.sigmoid(self.bn_atten(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))))
        feat = torch.mul(x, atten)
        x = x + feat
        return x

class FAM(nn.Module): 
    def __init__(self, in_channels):
        super(FAM, self).__init__()
        self.lateral_conv = FSM(in_channels)
        self.offset = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        self.bn_offset = nn.GroupNorm(num_groups=64, num_channels=in_channels)
        self.dcpack_l2 = DCNv2(in_channels, in_channels, 3, 1, 1, 8)
        nn.init.kaiming_uniform_(self.offset.weight, a=1)

    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        offset = self.bn_offset(self.offset(torch.cat([feat_arm, feat_up], dim=1)))  # concat for offset by compute the dif

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))  # [feat, offset]
        return feat_align + feat_arm

class oafpn(nn.Module):
    def __init__(self, bottom_up, layers_begin, layers_end):
        super(oafpn, self).__init__()
        assert layers_begin > 1 and layers_begin < 6
        assert layers_end > 4 and layers_begin < 8
        in_channels = [256, 512, 1024, 2048]  
        in_channels = in_channels[layers_begin-2:]

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        self.F5 = ECAG(in_channels[3])
        self.conv_F5_P5 = nn.Conv2d(in_channels[3], 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_F5_P5 = nn.GroupNorm(num_groups=64, num_channels=256)

        self.conv_F5_I = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_F5_I = nn.GroupNorm(num_groups=64, num_channels=256)


        self.feature_align = FAM(in_channels[3] // 8)

        self.ECAG = ECAG(in_channels=in_channels[3] // 8)

        self.NonLocal2D = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.Non_bn = nn.GroupNorm(num_groups=64, num_channels=256)

        self.conv_1x1_4 = nn.Conv2d(1024, 256, 1, bias=False)
        self.conv_1x1_3 = nn.Conv2d(512, 256, 1, bias=False)
        self.conv_1x1_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_1x1 = nn.Conv2d(512, 1024, 1, bias=False)

        self.bn_1x1_4 = nn.GroupNorm(num_groups=64, num_channels=256)
        self.bn_1x1_3 = nn.GroupNorm(num_groups=64, num_channels=256)
        self.bn_1x1_2 = nn.GroupNorm(num_groups=64, num_channels=256)

        self.conv_F3_1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_F2_1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_F4_1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn_F3_1 = nn.GroupNorm(num_groups=64, num_channels=256)
        self.bn_F2_1 = nn.GroupNorm(num_groups=64, num_channels=256)
        self.bn_F4_1 = nn.GroupNorm(num_groups=64, num_channels=256)

        self.conv_F3_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_F2_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_F4_2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn_F3_2 = nn.GroupNorm(num_groups=64, num_channels=256)
        self.bn_F2_2 = nn.GroupNorm(num_groups=64, num_channels=256)
        self.bn_F4_2 = nn.GroupNorm(num_groups=64, num_channels=256)

        self.conv_F3_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_F2_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.conv_F4_3 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.bn_F3_3 = nn.GroupNorm(num_groups=64, num_channels=256)
        self.bn_F2_3 = nn.GroupNorm(num_groups=64, num_channels=256)
        self.bn_F4_3 = nn.GroupNorm(num_groups=64, num_channels=256)

        self.bottom_up = bottom_up
        self.output_b = layers_begin
        self.output_e = layers_end

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        bottom_up_features = bottom_up_features[self.output_b - 2:]
        results = []
        l = bottom_up_features

        C2 = l[0]
        C3 = l[1]
        C4 = l[2]
        C5 = l[3]

        F5 = C5 * self.F5(C5) + C5
        P5 = F.relu_(self.bn_F5_P5(self.conv_F5_P5(F5)))
        F5_I1, F5_I2 = torch.chunk(F5, 2, 1)
        F5_I = F.relu_(self.bn_F5_I(self.conv_F5_I(self.pixel_shuffle(F5_I1) + self.pixel_shuffle(F5_I2))))

        C5_1,C5_2 = torch.chunk(C5, 2, 1)    
        F4 = F.relu_(self.bn_1x1_4(self.pixel_shuffle(C5_1) + self.pixel_shuffle(C5_2) + self.conv_1x1_4(C4)))
        F3 = F.relu_(self.bn_1x1_3(self.pixel_shuffle(C4) + self.conv_1x1_3(C3)))
        F2 = F.relu_(self.bn_1x1_2(self.pixel_shuffle(self.conv_1x1(C3))+self.conv_1x1_2(C2)))

        F3_1 = F.relu_(self.bn_F3_1(self.conv_F3_1(F3 + F.interpolate(F4, size=F3.shape[-2:], mode="nearest") + F.adaptive_max_pool2d(F2, output_size=F3.shape[-2:]))))

        F4_1 = F.relu_(self.bn_F4_1(self.conv_F4_1(F4 + F.adaptive_max_pool2d(F3_1, output_size=F4.shape[-2:]))))
        F2_1 = F.relu_(self.bn_F2_1(self.conv_F2_1(F2 + F.interpolate(F3_1, size=F2.shape[-2:], mode="nearest"))))

        F3_2 = F.relu_(self.bn_F3_2(self.conv_F3_2(F3_1 + F.interpolate(F4_1, size=F3_1.shape[-2:], mode="nearest") + F.adaptive_max_pool2d(F2_1, output_size=F3_1.shape[-2:]))))

        F4_2 = F.relu_(self.bn_F4_2(self.conv_F4_2(F4_1 + F.adaptive_max_pool2d(F3_2, output_size=F4_1.shape[-2:]))))
        F2_2 = F.relu_(self.bn_F2_2(self.conv_F2_2(F2_1 + F.interpolate(F3_2, size=F2_1.shape[-2:], mode="nearest"))))

        F3_3 = F.relu_(self.bn_F3_3(self.conv_F3_3(F3_2 + F.interpolate(F4_2, size=F3_2.shape[-2:], mode="nearest") + F.adaptive_max_pool2d(F2_2, output_size=F3_2.shape[-2:]))))

        F4_3 = F.relu_(self.bn_F4_3(self.conv_F4_3(F4_2 + F.adaptive_max_pool2d(F3_3, output_size=F4_2.shape[-2:]))))
        F2_3 = F.relu_(self.bn_F2_3(self.conv_F2_3(F2_2 + F.interpolate(F3_3, size=F2_2.shape[-2:], mode="nearest"))))

        P4 = F4_3
        P3 = F3_3
        P2 = F2_3

        out_size = P4.shape[-2:]

        I_P4 = F.interpolate(P4, size=out_size, mode="bilinear", align_corners=False)
        I_P3 = F.adaptive_max_pool2d(P3, output_size=out_size)
        I_P2 = F.adaptive_max_pool2d(P2, output_size=out_size)

        I = (I_P4 + I_P3 + I_P2) / 3
        I = self.feature_align(I, F5_I)

        I = F.relu_(self.Non_bn(self.NonLocal2D(I)))

        CA = self.ECAG(I)
        residual_R5 = F.adaptive_max_pool2d(I, output_size=C5.shape[-2:])

        R5 = residual_R5 * CA + P5
        residual_R4 = F.adaptive_max_pool2d(I, output_size=C4.shape[-2:])
        R4 = residual_R4 * CA + P4
        residual_R3 = F.interpolate(I, size=C3.shape[-2:], mode="bilinear", align_corners=False)
        R3 = residual_R3 * CA + P3
        residual_R2 = F.interpolate(I, size=C2.shape[-2:], mode="bilinear", align_corners=False)
        R2 = residual_R2 * CA + P2
        for i in [R5, R4, R3, R2]:
            results.append(i)

        if (self.output_e == 6):
            R6 = F.max_pool2d(results[0], kernel_size=1, stride=2, padding=0)
            results.insert(0, R6)

        return results