import torch
import torch.nn as nn

from einops import rearrange
from ultralytics.nn.modules.mobileViT import Transformer,conv_1x1_bn,conv_nxn_bn,prime_factors


class LocalTrans(nn.Module):
    def __init__(self, patch_size, ratio, mlp_dim, depth=3, dim=96, dropout=0.):
        super(LocalTrans, self).__init__()
        self.ph = patch_size
        self.pw = patch_size * ratio

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
    def forward(self,x):
        _, _, h, w = x.shape
        # val 不是1024x1024
        if h % self.ph != 0:
            self.ph = prime_factors(h)
        if w % self.pw != 0:
            self.pw = prime_factors(w)

        x = rearrange(x, 'b d (h ph) (w pw) -> b (h w) (ph pw) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (h w) (ph pw) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        return x


class GobalTrans(nn.Module):
    def __init__(self, patch_size, mlp_dim, depth=3, dim=96, dropout=0.):
        super(GobalTrans, self).__init__()
        self.ph = patch_size
        self.pw = patch_size

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

    def forward(self,x):
         # Global representations
        _, _, h, w = x.shape
        # val 不是1024x1024
        if h % self.ph != 0:
            self.ph = prime_factors(h)
        if w % self.pw != 0:
            self.pw = prime_factors(w)
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        return x


class DeformationViTBlock(nn.Module):
    def __init__(self, in_channel,out_channel, kernel_size, patch_size, ratio, mlp_dim, depth=3, dim=96, dropout=0.):
        super(DeformationViTBlock, self).__init__()
        self.ratio = ratio

        assert in_channel == out_channel

        self.conv1 = conv_nxn_bn(in_channel, 2 * in_channel, kernel_size)
        self.conv2 = conv_1x1_bn(in_channel, dim)
        self.conv3 = conv_1x1_bn(in_channel, dim)
        
        self.local_list = nn.ModuleList(
            LocalTrans(ratio=r,patch_size=patch_size,mlp_dim=mlp_dim,
                                        depth=depth,dropout=dropout) for r in self.ratio)
        
        self.gobal_trans = GobalTrans(patch_size=patch_size,mlp_dim=mlp_dim,depth=depth,dropout=dropout)


        self.conv4 = conv_1x1_bn((len(ratio)+1) * dim, out_channel)
        self.conv5 = conv_nxn_bn(2 * out_channel, out_channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()
        
        x = self.conv1(x)
        _,c,_,_ = x.shape
        x_gobal = x[:,:c//2,:,:]
        x_local = x[:,c//2:,:,:]
        x_local = self.conv2(x_local)
        x_gobal = self.conv3(x_gobal)
        
        x_locals = []
        for local in self.local_list:
            x_locals.append(local(x_local))

        x_gobal = self.gobal_trans(x_gobal)
        x = x_gobal.clone()

        #concat
        for i in x_locals:
            x = torch.cat((i,x), 1)
        
        # Fusion
        x = self.conv4(x)
        x = torch.cat((x, y), 1)
        x = self.conv5(x)
        return x
