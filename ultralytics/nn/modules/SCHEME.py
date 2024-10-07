import torch
import torch.nn as nn

#############注意力机制########################
class Attention(nn.Module):
    def __init__(self,embed_dim,num_heads,qkv_scale=None,dropout_rate=0.0,attention_dropout=0.0):
        super(Attention,self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        #详情见推导 head_dim是最终结果z的单头向量长度
        self.head_dim = int(embed_dim/num_heads)# 多头注意力 保持embed dim不变
        self.all_head_dim = self.head_dim*num_heads #防止整除不尽

        self.q = nn.Linear(in_features= embed_dim,
                          out_features= self.all_head_dim)
        self.k = nn.Linear(in_features= embed_dim,
                          out_features= self.all_head_dim)
        self.v = nn.Linear(in_features= embed_dim,
                          out_features= self.all_head_dim)
        # scale缩放的一步：sigmoid(s)/根号head_dim
        self.scale = self.head_dim ** -0.5 if qkv_scale is None else qkv_scale
        self.softmax = nn.Softmax(-1)
        # 最后将多头整合到一起
        self.proj = nn.Linear(in_features=self.all_head_dim,out_features=embed_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.attn_dropout = nn.Dropout(attention_dropout)


    def forward(self,q,k,v):
        batch_size,num_patch = q.shape[0],q.shape[1]
        
        q = self.q(q) # [batch_size num_patch all_head_dim]
        k = self.k(k)
        k = k.transpose(3,2)
        v = self.v(v)

        attn = torch.matmul(q,k) # q*kT 矩阵乘法
        attn = self.scale * attn
        attn = self.softmax(attn) #s 放缩
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn,v)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class MLP_block(nn.Module):
    def __init__(self,embed_dim,mlp_ratio=8.0,dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features=embed_dim,out_features=int(embed_dim*mlp_ratio))
        self.fc2 = nn.Linear(in_features=int(embed_dim*mlp_ratio),out_features=embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)
        return x

class BD_MLP(nn.Module):
    def __init__(self,dim,block_num,mlp_ratio=4.0,dropout_rate=0.0):
        super().__init__()
        self.divide_dim = dim//block_num
        self.mlp_block = nn.Sequential()
        for _ in range(block_num):
            self.mlp_block.append(MLP_block(self.divide_dim,mlp_ratio,dropout_rate))
        
    def forward(self,x):
        x_split = torch.split(x,self.divide_dim,-1) 
        i = 0
        y = []   
        for mlp in self.mlp_block:
            y.append(mlp(x_split[i]))
            i += 1
        x = torch.concat(y,-1)
        return x


class CCAttn(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.attn = Attention(embed_dim=embed_dim,num_heads=num_heads,qkv_scale=None)

    def forward(self,x):
        h = x.clone()
        x = self.attn(x,x,x)
        x = h + x
        return x
        
#相当于一个encoder中的MLP部分
class SCHEME(nn.Module):
    def __init__(self,embed_dim,a,block_num,mlp_ratio=8.0,dropout_rate=0.0,num_heads=4):
        super().__init__()
        self.alpha = a

        self.cca = CCAttn(embed_dim=embed_dim,num_heads=num_heads)
        self.bd_mlp = BD_MLP(embed_dim,block_num,mlp_ratio,dropout_rate)

    def forward(self,x):
        y = x.clone()
        x = self.bd_mlp(x)
        y = self.cca(y)
        out = x * self.alpha + y * (1 - self.alpha)
        return out
