import torch
import torch.nn as nn
from ultralytics.nn.modules.SCHEME import SCHEME

class Potential_Block(nn.Module):
    def __init__(self,in_ch,o_ch,num_heads = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = int(o_ch/num_heads)# 多头注意力 保持embed dim不变

        self.conv1 = nn.Conv2d(in_ch,o_ch,kernel_size=1,stride=1)
        self.norm = nn.SyncBatchNorm(o_ch)
        self.act = nn.Mish()
        
        self.conv2 = nn.ModuleList([nn.Conv2d(self.head_dim,self.head_dim,kernel_size=3,stride=1,padding=1) for _ in range(self.num_heads)])
    def forward(self,x):
        x = self.act(self.norm(self.conv1(x)))
        list_x = torch.chunk(x, self.num_heads, dim=1)
        list_out = []
        for i in range(self.num_heads):
            list_out.append(self.conv2[i](list_x[i]))
        x = torch.cat(list_out, 1)
        return x

class CSAM(nn.Module):
    def __init__(self,in_ch,o_ch,num_heads = 4):
        super().__init__()

        self.q = nn.Linear(in_features=in_ch,out_features=o_ch)
        self.k = nn.Linear(in_features=in_ch,out_features=o_ch)
        self.v = nn.Linear(in_features=in_ch,out_features=o_ch)
        
        self.potential_block1 = Potential_Block(o_ch,o_ch,num_heads)
        self.potential_block2 = Potential_Block(o_ch,o_ch,num_heads)
        self.potential_block3 = Potential_Block(o_ch,o_ch,num_heads)
        self.softmax = nn.Sigmoid()
        self.act = nn.Mish()
        self.norm = nn.SyncBatchNorm(o_ch)

        self.conv3 = nn.Conv2d(o_ch,o_ch,kernel_size=1,stride=1)
        self.conv4 = nn.Conv2d(o_ch,o_ch,kernel_size=3,stride=1,padding=1)
    def forward(self,x):
        h = x.clone()
        
        x = x.permute(0,2,3,1)
        q = self.q(x).permute(0,3,1,2)
        k = self.k(x).permute(0,3,1,2)
        v = self.v(x).permute(0,3,1,2)
        x = x.permute(0,3,1,2)

        q = self.potential_block1(q)
        k = self.potential_block2(k)
        v = self.potential_block3(v)
        
        attn = self.softmax(q * k)
        attn = attn * (self.act(self.norm(v)))

        x = self.act(self.conv3(attn))
        x = self.act(self.conv4(x))
        x = h + x
        return x

class Former_BottleNeck(nn.Module):
    def __init__(self,in_ch,o_ch,num_heads = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,o_ch,kernel_size=1,stride=1)

        self.csam = CSAM(o_ch, o_ch,num_heads)
        self.norm = nn.SyncBatchNorm(o_ch)

        self.conv2 = nn.Conv2d(o_ch,o_ch//2,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(o_ch//2,o_ch,kernel_size=3,stride=1,padding=1)

        self.act = nn.Mish()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.act(x)
        
        h = x.clone()
        x = self.norm(x)
        x = self.csam(x)
        x = self.norm(x)
        x = h + x
        

        h = x.clone()
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = h + x

        x = self.norm(x)
        return x
    
class Former_BottleNeck_SCHEME(nn.Module):
    def __init__(self,in_ch,o_ch,num_heads = 4,alpha=0.7,block_num=8):
        super().__init__()
        self.alpha = alpha
        self.conv1 = nn.Conv2d(in_ch,o_ch,kernel_size=1,stride=1)

        self.csam = CSAM(o_ch, o_ch,num_heads)
        self.norm = nn.SyncBatchNorm(o_ch)

        self.mlp = SCHEME(o_ch,a=alpha,block_num=block_num)

        self.act = nn.Mish()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.norm(x)

        h = x.clone()
        x = self.csam(x)
        x = h + x
        x = self.norm(x)

        h = x.clone()
        x = x.permute(0,2,3,1)
        x = self.act(self.mlp(x))
        x = x.permute(0,3,1,2)
        x = h + x

        x = self.norm(x)
        return x
    
    def infer_mode(self):
        self.alpha = 1