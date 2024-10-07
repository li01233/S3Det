import torch
import torch.nn as nn

from einops import rearrange
from ultralytics.nn.modules.mobileViT import conv_1x1_bn,conv_nxn_bn,prime_factors,PreNorm,Attention
from ultralytics.nn.modules.SCHEME import SCHEME


class SCHEME_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, alpha,block_num,mlp_ratio,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, SCHEME(dim, alpha,block_num,mlp_ratio,dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SCHEME_MobileViTBlock(nn.Module):
    def __init__(self, in_channel,out_channel, kernel_size, patch_size,
                alpha=0.7,block_num=8,mlp_ratio=8.0, depth=3, dim=96, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.alpha = alpha

        self.conv1 = conv_nxn_bn(in_channel, out_channel, kernel_size)
        self.conv2 = conv_1x1_bn(out_channel, dim)

        self.transformer = SCHEME_Transformer(dim, depth, 4, 8, self.alpha, block_num, mlp_ratio, dropout)

        self.conv3 = conv_1x1_bn(dim, out_channel)
        self.conv4 = conv_nxn_bn(2 * out_channel, out_channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
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

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x
    
    def infer_mode(self):
        self.alpha = 1