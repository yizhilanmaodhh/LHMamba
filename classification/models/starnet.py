"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from typing import List
import antialiased_cnns
from torch import Tensor
import torch.nn.functional as F

from basicsr.models.archs.LoFormer_arch import TransformerBlock_2b,FreqBranch,TransformerBlock
from EfficientViM.EfficientViM.classification.models.EfficientViM import HSMSSD

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


@register_model
def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


# very small networks #
@register_model
def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)


@register_model
def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)


@register_model
def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)



class PA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim*4, 1, bias=False),
            norm_layer(dim*4),
            act_layer(),
            nn.Conv2d(dim*4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        att = self.p_conv(x)
        x = x * self.gate_fn(att)

        return x


class LA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MRA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.max_m1(x)
        x_tem = self.max_m2(x_tem)
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))

        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)

        out = x[:, :self.channel, :, :] * F.interpolate(self.gate_fn(att),
                                                        size=(x.shape[-2], x.shape[-1]),
                                                        mode='nearest')
        return out

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


class GA12(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x_ = self.proj_1(x_)
        x_ = self.activation(x_)
        attn1 = self.conv0(x_)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        x_ = x_ * attn
        x_ = self.proj_2(x_)
        x = self.uppool(x_, indices=idx)
        return x


class D_GA(nn.Module):

    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = GA(dim)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)

    def forward(self, x):
        x_, idx = self.downpool(x)
        x = self.norm(self.attn(x_))
        x = self.uppool(x, indices=idx)

        return x


class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Block1(nn.Module):
    def __init__(self, dim, norm_layer, act_layer, att_kernel, stage, mlp_ratio=3, drop_path=0.,):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 2
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.PA = PA(self.dim_split, norm_layer, act_layer)  # PA is point attention
        # self.LA = LA(self.dim_split, norm_layer, act_layer)  # LA is local attention
        # self.MRA = MRA(self.dim_split, att_kernel, norm_layer)  # MRA is medium-range attention

        # mlp_hidden_dim = int(dim * mlp_ratio)

        # mlp_layer: List[nn.Module] = [
        #     nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
        #     norm_layer(mlp_hidden_dim),
        #     act_layer(),
        #     nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        # ]
        self.norm1 = norm_layer(dim)

        # 注意力模块
        if stage == 1:
            self.attn = GA12(self.dim_split, act_layer)         # GA12 is global attention (stages of 1 and 2)
        elif stage == 2:
            self.attn = D_GA(self.dim_split, norm_layer)         # GA3 is global attention (stage of 3)
        elif stage == 3:
            self.attn = GA(self.dim_split)
            self.norm = norm_layer(self.dim_split)
        elif stage == 4:
            self.attn = GA(self.dim_split)                     # GA4 is global attention (stage of 4)
            self.norm = norm_layer(self.dim_split)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # 按通道分割
        x1, x2 = torch.split(x, [self.dim_split, self.dim_split], dim=1)
        # 原 StarNet 的操作
        x1_original = self.dwconv(x1)
        x1_f1, x1_f2 = self.f1(x1_original), self.f2(x1_original)
        x1 = self.act(x1_f1) * x1_f2
        x1 = self.dwconv2(self.g(x1))
        # 注意力模块操作
        x2 = self.attn(x2)
        # 合并特征
        x = torch.cat([x1, x2], dim=1)
        x = input + self.drop_path(x)
        # x = self.dwconv(x)
        # x1, x2 = self.f1(x), self.f2(x)
        # x = self.act(x1) * x2
        # x = self.dwconv2(self.g(x))
        # x = input + self.drop_path(x)
        return x


class Block2(nn.Module):
    def __init__(self, dim, norm_layer, act_layer, att_kernel, stage,mlp_ratio=3, drop_path=0.,ssd_expand=1, state_dim=64):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.HSMSSD = HSMSSD(d_model=mlp_ratio * dim, ssd_expand=ssd_expand,state_dim=state_dim)

        # self.PA = PA(self.dim_split, norm_layer, act_layer)  # PA is point attention
        self.LA = LA(self.dim_split, norm_layer, act_layer)  # LA is local attention
        # self.MRA = MRA(self.dim_split, att_kernel, norm_layer)  # MRA is medium-range attention

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.norm1 = norm_layer(dim)

        # 注意力模块
        if stage == 1:
            self.attn = GA12(self.dim_split, act_layer)         # GA12 is global attention (stages of 1 and 2)
        elif stage == 2:
            self.attn = D_GA(self.dim_split, norm_layer)         # GA3 is global attention (stage of 3)
        elif stage == 3:
            self.attn = GA(self.dim_split)
            self.norm = norm_layer(self.dim_split)
        elif stage == 4:
            self.attn = GA(self.dim_split)                     # GA4 is global attention (stage of 4)
            self.norm = norm_layer(self.dim_split)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # 按通道分割
        x1, x2, x3, x4 = torch.split(x, [self.dim_split, self.dim_split, self.dim_split, self.dim_split], dim=1)
        # 原 StarNet 的操作
        x1_original = self.dwconv(x1)
        x1_f1, x1_f2 = self.f1(x1_original), self.f2(x1_original)
        x1 = self.act(x1_f1) * x1_f2
        x1 = self.dwconv2(self.g(x1))
        x2 = self.LA(x2)
        x3, h = self.HSMSSD(x3.flatten(2))
        # 注意力模块操作
        x4 = self.attn(x4)
        # 合并特征
        x = torch.cat((x1, x2, x3, x4), 1)
        x = input + self.drop_path(x)
        # x = self.dwconv(x)
        # x1, x2 = self.f1(x), self.f2(x)
        # x = self.act(x1) * x2
        # x = self.dwconv2(self.g(x))
        # x = input + self.drop_path(x)
        return x


class StarNetA1(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=30, att_kernel= 11, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block1(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)
class StarNetB1(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block2(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)

@register_model
def starnet_A1(pretrained=False, **kwargs):
    model = StarNetA1(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model

@register_model
def starnet_B1(pretrained=False, **kwargs):
    model = StarNetB1(32, [1, 1, 4, 2] , att_kernel=(11, 11, 11, 11), **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model
