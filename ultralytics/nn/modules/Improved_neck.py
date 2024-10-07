import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ultralytics.nn.modules.Improved_neck_layers import Conv,onnx_AdaptiveAvgPool2d,get_shape,trans_Block,SimConv,RepVGGBlock,h_sigmoid,get_avg_pool
from mmcv.cnn import ConvModule
__all__ = ('L_FAM', 'L_IFM', 'Split', 'SimConv', 'L_LAF', 'Inject', 'RepBlock', 'H_FAM', 'H_IFM', 'H_LAF')

class L_FAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        _, _, H, W = x_s.shape
        output_size = [H, W]

        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d

        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)

        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out

class L_IFM(nn.Module):
    def __init__(self, in_channels, embed_dims, fuse_block_num, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, embed_dims, kernel_size=1, stride=1, padding=0)
        self.block = nn.ModuleList([RepVGGBlock(embed_dims, embed_dims) for _ in range(fuse_block_num)]) if fuse_block_num > 0 else nn.Identity
        self.conv2 = Conv(embed_dims, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        for b in self.block:
            x = b(x)
        out = self.conv2(x)
        return out

class H_FAM(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
    
    def forward(self, inputs):
        _, _, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        
        # output_size = np.array([H, W])
        output_size = [H, W]
       
        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d
        
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        
        out = [self.pool(inp, output_size) for inp in inputs]
        
        return torch.cat(out, dim=1)

#由一堆transformer组成
class H_IFM(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(trans_Block(
                    embedding_dim, key_dim=key_dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                    drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg, act_layer=act_layer))
    
    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

class L_LAF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channels, out_channels, 1, 1)
        self.cv_fuse = SimConv(out_channels * 3, out_channels, 1, 1)
        self.downsample = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        _, _, H, W = x[1].shape
        output_size = (H, W)
        
        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])
        
        x0 = self.downsample(x[0], output_size)
        x1 = self.cv1(x[1])
        x2 = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))

class H_LAF(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d
        
        _, _, H, W = x[1].shape
        output_size = np.array([H, W])
        x1 = self.pool(x[0], output_size)
        
        return torch.cat([x1, x[1]], 1)

class Inject(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_index: int,
            split_idx: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
            global_inp=None,
    ) -> None:
        super().__init__()
        self.norm_cfg = norm_cfg
        self.global_index = global_index
        self.split_idx = split_idx
        
        if not global_inp:
            global_inp = inp
        
        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(global_inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()
    
    def forward(self, x):
        '''
        x_g: global features
        x_l: local features
        '''
        x_g = x[self.global_index]
        x_l = x[1 - self.global_index]
        if(type(x_g) == tuple):
            x_g = x_g[self.split_idx]
        if(type(x_l) == tuple):
            x_l = x_l[self.split_idx]

        _, _, H, W = x_l.shape
        _, _, g_H, g_W = x_g.shape
        use_pool = H < g_H
        
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)
        
        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])
            
            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)
        
        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act + global_feat
        return out

class Split(nn.Module):
    def __init__(self, trans_channels):
        super().__init__()
        self.trans_channels = trans_channels

    def forward(self, x):
        return x.split(self.trans_channels, dim=1)
    

