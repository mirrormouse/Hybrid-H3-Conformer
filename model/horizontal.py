import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from H3.src.models.ssm.h3 import H3 as H3Module


import torch.nn.init as init
from torch import Tensor
from typing import Tuple

from model.conformer import FeedForward, PreNorm, ConformerConvModule, Scale, Conv2dSubampling, Attention, Linear


class H3Conformer_HybridBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        attn_dim,
        dim_head=64,
        h3_dim_head=8,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.,
        ff_dropout=0.,
        conv_dropout=0.,
        conv_causal=True
    ):
        super().__init__()

        self.attn_dim = attn_dim
        self.h3_dim = dim - self.attn_dim

        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        # 分割した入力のそれぞれを処理する2つのレイヤーを定義
        self.attn = PreNorm(self.attn_dim, Attention(dim=self.attn_dim, dim_head=dim_head, heads=heads, dropout=attn_dropout))
        self.h3 = PreNorm(self.h3_dim, H3Module(d_model=self.h3_dim, head_dim=h3_dim_head))
        self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.conv = ConformerConvModule(dim=dim, causal=conv_causal, expansion_factor=conv_expansion_factor, kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)


        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.ff3 = Scale(0.5, PreNorm(dim, self.ff3))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x

        # 入力を2分割
        x1, x2 = x.split([self.attn_dim, self.h3_dim], dim=-1)

        # 分割した入力をそれぞれの層に通す
        x1 = self.attn(x1)
        x2 = self.h3(x2)

        # 出力をconcatenate
        x_main = torch.cat([x1, x2], dim=-1)
        x_main = self.ff3(x_main) + x_main
        x = x_main + x

        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x




class Horizontal_CH4(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        depth,
        dim,
        attn_dim,
        *,
        h3_dim_head,
        attn_layer_idx,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 15,
        attn_dropout = 0.1,
        ff_dropout = 0.1,
        conv_dropout = 0.1,
        conv_causal = True
    ):
        dim_head = int(dim / heads)
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        
        input_dropout_p = 0.1
        self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=dim)
        self.input_projection = nn.Sequential(
            Linear(dim * (((input_dim - 1) // 2 - 1) // 2), dim),
            nn.Dropout(p=input_dropout_p),
        )

        for idx in range(depth):
            self.layers.append(H3Conformer_HybridBlock(
                dim = dim,
                attn_dim = attn_dim,
                dim_head = dim_head,
                h3_dim_head = h3_dim_head,
                heads = heads,
                ff_mult = ff_mult,
                conv_expansion_factor = conv_expansion_factor,
                conv_kernel_size = conv_kernel_size,
                conv_causal = conv_causal,
            ))

        self.fc = Linear(dim, output_dim, bias=False)

        

    def forward(self, input, input_lengths):

        outputs, output_lengths = self.conv_subsample(input, input_lengths)
        outputs = self.input_projection(outputs)

        for block in self.layers:
            outputs = block(outputs)

        outputs = self.fc(outputs)
        outputs = nn.functional.log_softmax(outputs, dim=-1)


        return outputs, output_lengths