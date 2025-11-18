import torch
import torch.nn as nn
from timm.models.vision_transformer import DropPath, Mlp
from models.TFMamba import ConcatMambaFusionBlock, CrossMambaFusionBlock


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        (B, N1, C), N2 = x1.shape, x2.shape[1]
        q = self.q(x1).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x1 = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)
        return x1, attn




class CrossAttention_DenseAVInteractions(nn.Module):
    def __init__(self, in_scales, in_channels, stages):
        super().__init__()
        self.Avgpool1 = nn.AdaptiveAvgPool2d(8)
        self.Avgpool2 = nn.AdaptiveAvgPool2d(6)
        self.Avgpool3 = nn.AdaptiveAvgPool2d(4)
        self.Maxpool1 = nn.AdaptiveMaxPool2d(8)
        self.Maxpool2 = nn.AdaptiveMaxPool2d(6)
        self.Maxpool3 = nn.AdaptiveMaxPool2d(4)

        self.upsample = nn.Upsample(size=in_scales, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, 256, 1)
        self.stages = stages
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.stages == ['layer2']:
            self.TFMamba = CrossMambaFusionBlock(hidden_dim=96, mlp_ratio=0.0, d_state=8, )
        elif self.stages == ['layer3']:
            self.TFMamba = CrossMambaFusionBlock(hidden_dim=48, mlp_ratio=0.0, d_state=8, )
        elif self.stages == ['layer4']:
            self.TFMamba = CrossMambaFusionBlock(hidden_dim=24, mlp_ratio=0.0, d_state=8, )

    def forward(self, xv, xa):
        nv, na = xv.shape[1], xa.shape[1]

        xv, xa = self.TFMamba(xv, xa)
        if self.stages == ['layer2']:
            xv = self.Avgpool1(xv)
            xa = self.Maxpool1(xa)
        elif self.stages == ['layer3']:
            xv = self.Avgpool2(xv)
            xa = self.Maxpool2(xa)
        elif self.stages == ['layer4']:
            xv = self.Avgpool3(xv)
            xa = self.Maxpool3(xa)

        B, C, H, W = xv.shape

        xva = torch.cat((
            xv.unsqueeze(4).unsqueeze(5).expand(B, C, H, W, H, W),
            xa.unsqueeze(2).unsqueeze(3).expand(B, C, H, W, H, W)
        ), dim=1)

        xva = xva.reshape(xva.shape[0], xva.shape[1], xva.shape[2] * xva.shape[3], xva.shape[4] * xva.shape[5])
        xva = self.upsample(xva)
        xva = self.conv1(xva)

        return xva


class FusionBlock_DenseAVInteractions(nn.Module):
    def __init__(self, dim, num_heads, attn_ratio=0.25, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_mm = norm_layer(dim)
        self.norm1_aud = norm_layer(dim)
        self.norm1_img = norm_layer(dim)
        self.attn = CrossAttention_DenseAVInteractions(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, dim_ratio=attn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xmm, xv, xa, return_attention=False):
        xmm, xv, xa = self.norm1_mm(xmm), self.norm1_img(xv), self.norm1_aud(xa)
        res_fusion, attn = self.attn(xmm, xv, xa)
        xmm = xmm + self.drop_path(res_fusion)
        if return_attention:
            return attn

        res_fusion = self.mlp(self.norm2(xmm))
        xmm = xmm + self.drop_path(res_fusion)
        return xmm