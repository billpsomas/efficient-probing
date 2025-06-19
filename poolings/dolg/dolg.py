#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Written by feymanpriv

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import poolings.dolg.net as net

""" Dolg models """

class SpatialAttention2d(nn.Module):
    '''
    SpatialAttention2d
    2-layer 1x1 conv network with softplus activation.
    '''
    def __init__(self, in_c, s3_dim=1024, act_fn='relu', with_aspp=False, bn_eps=1e-5, bn_nom=0.1):
        super(SpatialAttention2d, self).__init__()
        
        self.with_aspp = with_aspp
        if self.with_aspp:
            self.aspp = ASPP(s3_dim)
        self.conv1 = nn.Conv2d(in_c, s3_dim, 1, 1)
        self.bn = nn.BatchNorm2d(s3_dim, eps=bn_eps, momentum=bn_nom)
        if act_fn.lower() in ['relu']:
            self.act1 = nn.ReLU()
        elif act_fn.lower() in ['leakyrelu', 'leaky', 'leaky_relu']:
            self.act1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(s3_dim, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

        for conv in [self.conv1, self.conv2]: 
            conv.apply(net.init_weights)

    def forward(self, x, cls=None, block_attmaps=None, return_attn=False):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        # Reshape from (b, c, h*w) to (b, c, h, w) if necessary
        x = x.permute(0, 2, 1).contiguous()
        b, c, hw = x.size()
        h = w = int(hw ** 0.5)  # assumes square dimensions
        x = x.view(b, c, h, w).contiguous()

        if self.with_aspp:
            x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        x = att * feature_map_norm
        # GAP on DOLG outputs
        if return_attn:
            return x.view(b, c, -1).permute(0, 2, 1).mean(1), att_score
        return x.view(b, c, -1).permute(0, 2, 1).mean(1)
    
    def __repr__(self):
        return self.__class__.__name__


class ASPP(nn.Module):
    '''
    Atrous Spatial Pyramid Pooling Module 
    '''
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp = []
        self.aspp.append(nn.Conv2d(in_c, 512, 1, 1))

        for dilation in [6, 12, 18]:
            _padding = (dilation * 3 - dilation) // 2
            self.aspp.append(nn.Conv2d(in_c, 512, 3, 1, padding=_padding, dilation=dilation))
        self.aspp = nn.ModuleList(self.aspp)

        self.im_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(in_c, 512, 1, 1),
                                     nn.ReLU())
        conv_after_dim = 512 * (len(self.aspp)+1)
        #FIXME: Dionysis: changed 2nd dim from 1024 to "in_c" to reduce capacity
        self.conv_after = nn.Sequential(nn.Conv2d(conv_after_dim, in_c, 1, 1), nn.ReLU())
        
        for dilation_conv in self.aspp:
            dilation_conv.apply(net.init_weights)
        for model in self.im_pool:
            if isinstance(model, nn.Conv2d):
                model.apply(net.init_weights)
        for model in self.conv_after:
            if isinstance(model, nn.Conv2d):
                model.apply(net.init_weights)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        aspp_out = [F.interpolate(self.im_pool(x), scale_factor=(h,w), mode="bilinear", align_corners=False)]
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))
        aspp_out = torch.cat(aspp_out, 1)
        x = self.conv_after(aspp_out)
        return x