import torch
from torch import nn

class stem(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, stride=1, groups=1):
        super(stem, self).__init__()
        padding = (kernel_size-1) // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.1)
        self.act = nn.SiLU(True)

    def forward(self, x):
        #print(x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()

    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class SqueezeExcite_efficientv2(nn.Module):
    def __init__(self, c1, c2, se_ratio=0.25, act_layer=nn.ReLU):
        super(SqueezeExcite_efficientv2, self).__init__()
        self.gate_fn = nn.Sigmoid()
        reduced_chs = int(c1 * se_ratio)
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(c1, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, c2, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x_se = self.gate_fn(x_se)
        x = x * (x_se.expand_as(x))
        return x

class FusedMBConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, se_ration=0, dropout_rate=0.2, drop_connect_rate=0.2):
        super().__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        self.has_expansion = expansion != 1
        expanded_c = c1 * expansion
        if self.has_expansion:
            self.expansion_conv = stem(c1, expanded_c, kernel_size=k, stride=s)
            self.project_conv = stem(expanded_c, c2, kernel_size=1, stride=1)
        else:
            self.project_conv = stem(c1, c2, kernel_size=k, stride=s)
        self.drop_connect_rate = dropout_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):
        if self.has_expansion:
            result = self.expansion_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)
        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                result = self.dropout(result)
            result += x

        return result

class MBConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, expansion=1, se_ration=0, dropout_rate=0.2, drop_connect_rate=0.2):
        super(MBConv, self).__init__()
        self.has_shortcut = (s == 1 and c1 == c2)
        expanded_c = c1 * expansion
        self.expansion_conv = stem(c1, expanded_c, kernel_size=1, stride=1)
        self.dw_conv = stem(expanded_c, expanded_c, kernel_size=k, stride=s, groups=expanded_c)
        self.se = SqueezeExcite_efficientv2(expanded_c, expanded_c, se_ration) if se_ration > 0 else nn.Identity()
        self.project_conv = stem(expanded_c, c2, kernel_size=1, stride=1)
        self.drop_connect_rate = drop_connect_rate
        if self.has_shortcut and drop_connect_rate > 0:
            self.dropout = DropPath(drop_connect_rate)

    def forward(self, x):
        result = self.expansion_conv(x)
        result = self.dw_conv(result)
        result = self.se(result)
        result = self.project_conv(result)
        if self.has_shortcut:
            if self.drop_connect_rate > 0:
                result = self.dropout(result)
            result += x
        return result