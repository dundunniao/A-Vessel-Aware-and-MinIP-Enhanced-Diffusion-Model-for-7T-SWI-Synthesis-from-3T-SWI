# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu
import torch.nn as nn
import math


#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.用于初始化权重和偏差的统一例程

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.全连接层

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)#前一个**代表调参，后一个*代表乘法。
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))

        return x

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).注意力权重计算，即softmax（Q^T * K）。
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.使用 FP32精度进行所有计算，但使用原始数据类型进行输入/输出/梯度以节省内存。

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):#张量 q 和 k，其中 q 代表查询（queries），k 代表键（keys）。
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        #使用 torch.einsum 计算查询和键的点积，并立即转换为FP32精度进行计算。将键 k 除以其维度的平方根来进行缩放（这是注意力机制中的常见做法以避免过大的数值）。
        # 计算softmax函数以得到注意力权重，计算完成后将结果转换回原来的数据类型 q.dtype。
        ctx.save_for_backward(q, k, w)#使用 ctx.save_for_backward 保存输入和输出，以便在反向传播时使用。
        return w #输出：注意力权重张量 w。

    @staticmethod
    def backward(ctx, dw): #输入：梯度张量 dw，对应于前向传播输出的梯度。
        q, k, w = ctx.saved_tensors #从保存的张量中恢复 q, k, 和 w。
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        #使用 torch._softmax_backward_data 计算softmax的梯度。这里同样首先将梯度转换为FP32。
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])#计算关于查询 q 和键 k 的梯度，使用 torch.einsum 进行有效的张量乘法，并按需缩放和转换数据类型。
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])#对Q求导，就是W的导数张量乘以K张量（W张量本身就是由K,Q相乘然后Softmax得到的）。
        return dq, dk #输出：梯度 dq 和 dk 分别对应于输入的 q 和 k。
    #通过使用FP32进行中间计算，然后再将结果转换回原始数据类型，这种方法可以在保持高精度计算的同时节省内存。

#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.构造函数 __init__ 接受多个参数，包括输入通道数 in_channels、输出通道数 out_channels、
# 嵌入通道数 emb_channels，以及一些控制模块结构的标志，如 up、down、attention 等。其中，up 和 down 控制上采样和下采样，attention 控制是否使用注意力机制。
# 此外，还有一些与注意力机制相关的参数，如 num_heads、channels_per_head 等。

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ): # emb_channels=64, num_heads=1, adaptive_scale=False
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        # self.num_heads绝大多数情况下为0
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
#norm0 是输入数据的 GroupNorm 归一化层。conv0 是一个卷积层，用于进行特征提取和变换。affine 是一个全连接层，用于计算注意力权重。norm1 是输出数据的 GroupNorm 归一化层。
#conv1 是另一个卷积层，用于进一步特征提取和变换。skip 是跳跃连接层，用于实现 U-Net 结构中的跳跃连接。根据条件是否需要进行上采样、下采样或通道数变换，选择不同的卷积核大小和操作。
#如果模块使用了注意力机制，则定义了额外的注意力计算相关的层，如 norm2、qkv 和 proj。总体而言，这段代码定义了一个 U-Net 网络中的基本模块，并通过一系列卷积层、归一化层和全连接层实现了特征提取、跳跃连接和注意力计算等功能。
#只是定义了以下模块，并没有串联起来。
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):#这段代码是 UNetBlock 类的前向传播函数 forward。它接受两个参数：x 是输入张量，emb 参考SongUNet模块，是噪声程度t。
        orig = x
        x = self.conv0(silu(self.norm0(x)))#对输入张量 x 进行一系列操作，通过 norm0 对输入数据进行归一化；经过 silu 激活函数处理后，再通过 conv0 进行卷积操作。

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)#接着，根据嵌入向量 emb 计算参数 params，并根据条件选择不同的处理方式：
        if self.adaptive_scale: # 不走这里
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else: # 走这里
            x = silu(self.norm1(x.add_(params)))
        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads: # 绝大多数情况下不走这里。走这里的时候一般是在解码器那里。

            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale

        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.DDPM++ 和 ADM 架构中使用的时间步长嵌入


@persistence.persistent_class
class LayerNorm(nn.Module):
    def __init__(self, hidden_size=None, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        # super(LayerNorm, self).__init__()
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias



@persistence.persistent_class
class SelfAttention(nn.Module):
    # def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
    def __init__(self, num_attention_heads=None, input_size=None, hidden_size=None, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1):
        # num_attention_heads：注意力头的数量。在多头注意力中，输入的表示会被分成多个头来进行并行计算。
        # hidden_dropout_prob：在输出部分应用的 dropout 比例，防止过拟合。
        print(f"Initializing SelfAttention with num_attention_heads={num_attention_heads}, input_size={input_size}, hidden_size={hidden_size}")
        super().__init__()
        # super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 确保 hidden_size 是 num_attention_heads 的倍数。如果不是，会抛出一个错误。因为每个头的维度是 hidden_size // num_attention_heads，因此必须能够整除。
        self.num_attention_heads = num_attention_heads # self.num_attention_heads：存储注意力头的数量。
        self.attention_head_size = int(hidden_size / num_attention_heads) # self.attention_head_size：每个注意力头的大小，等于 hidden_size // num_attention_heads。
        self.all_head_size = hidden_size # 所有注意力头的总大小，等于 hidden_size。

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob) # attn_dropout：对注意力概率应用的 dropout 层，用于防止过拟合。

        self.dense = nn.Linear(hidden_size, hidden_size) # 用于对上下文表示（经过注意力计算后）进行线性变换，生成新的隐藏状态。
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12) # 层归一化层，用于稳定训练并帮助加速收敛。
        self.out_dropout = nn.Dropout(hidden_dropout_prob) # 在最终输出应用的 dropout 层。

    def transpose_for_scores(self, x): # 用于将输入张量 x 重塑并转置，以便可以处理多头注意力机制中的多个注意力头。
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # x.size()[:-1]：获取除最后一个维度外的所有维度（通常是 batch size 和序列长度）。
        x = x.contiguous().view(*new_x_shape) # 通过 view 方法将最后一个维度分割成多个头和每个头的大小。
        return x.permute(0, 2, 1, 3) # 转置操作，排列维度为 [batch_size, num_attention_heads, seq_len, attention_head_size]，以便计算注意力。

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor) # mixed_query_layer、mixed_key_layer 和 mixed_value_layer 是通过线性变换得到的查询、键和值的表示。

        query_layer = self.transpose_for_scores(mixed_query_layer) # 将查询、键和值分别通过 transpose_for_scores 方法转置，以便处理多头注意力。
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # 计算注意力分数，即查询和键的点积。这里用 transpose(-1, -2) 来转置 key_layer，使得它与 query_layer 的形状匹配，以进行矩阵乘法。

        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # 对注意力分数进行缩放，防止在点积计算时数值过大，导致梯度爆炸。这个操作是 Transformer 中的标准做法。
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs) # 对注意力概率进行 dropout，防止过拟合。
        # 确保在推理时关闭Dropout层（通过model.eval()），否则它会影响结果。
        context_layer = torch.matmul(attention_probs, value_layer) # 通过加权求和计算上下文表示，即将注意力概率与值（value_layer）相乘，得到加权的上下文表示。
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # 将 context_layer 的维度转置回 [batch_size, seq_len, num_attention_heads, attention_head_size]。
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) # 重新调整 context_layer 的形状，将每个头的表示合并到一个大的表示中，得到最终的上下文表示。
        hidden_states = self.dense(context_layer) # 对上下文表示应用一个线性层（dense），生成新的隐藏状态。
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor) # 对隐藏状态和输入张量做加法，并应用层归一化。这是 Transformer 中的标准做法，确保输入的残差信息能够流动，避免梯度消失。
        # print('hidden_states', hidden_states.size()) # hidden_states torch.Size([1, 784, 512])

        return hidden_states # 输出的 hidden_states 仍然是 [1, 37440, 128],即输入输出形状不变。


@persistence.persistent_class
class Cross_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim): # in_dim 是 x_0 和 x_1 张量的通道数（即它们的 C 维度），因此它们的输入 x_0 和 x_1 的通道数应该与 in_dim 一致。
        # super(Cross_Attn, self).__init__()
        super().__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # 注意：这里的卷积层是 1x1 的卷积，即每个像素位置的特征图会通过卷积层进行通道数的转换，而空间维度（宽度和高度）保持不变。
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
    # x_0 guide x_1
    def forward(self, x_0, x_1):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self self.attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # x_0 和 x_1：这两个输入分别表示输入特征图。x_0 和 x_1 的维度都是 (B, C, W, H)
        batch_0, C_0, width_0, height_0 = x_0.size()
        batch_1, C_1, width_1, height_1 = x_1.size()
        # print("x_0.shape, x_1.shape:", x_0.shape, x_1.shape)
        proj_query = self.query_conv(x_0).view(batch_0, -1).permute(1, 0) # 对 x_0 输入应用卷积 query_conv，输出的维度为 (B, C/4, W, H)。
        # 然后通过 .view(batch_0, -1) 将其展平为二维矩阵，维度变为 (B, C/4 * W * H)。使用 .permute(1, 0) 转置该矩阵，变成 (C/4 * W * H, B) 的形状，方便后续矩阵乘法操作。

        proj_key = self.key_conv(x_1).view(batch_1, -1)  # 对 x_1 输入应用卷积 key_conv，同样会得到形状为 (B, C/4, W, H) 的张量，然后展平为二维矩阵，形状为 (B, C/4 * W * H)。
        energy = torch.mm(proj_key, proj_query)
        # proj_key 和 proj_query 之间执行矩阵乘法，得到一个 energy 矩阵，它表示了每个位置之间的相似度（注意力分数）。其形状为 (B, B)。
        attention = self.softmax(energy)  # the shape are K_number * N_number 使用 softmax 对 energy 进行归一化，得到一个注意力权重矩阵，形状为 (B, B)。
        proj_value = self.value_conv(x_1).view(batch_1, -1)
        # 对 x_0 输入应用卷积 value_conv，输出的维度为 (B, C, W, H)，然后展平为二维矩阵 (B, C * W * H)，得到QKV里面的V。

        out = torch.mm(attention, proj_value)     # (K_number * N_number) * (N_number*(512*8*8)) output a tensor, the shape is K_number*(512*8*8)
        out = out.view(batch_0, C_0, width_0, height_0)

        out = self.gamma * out + x_0     # (K_number * 512 * 8 * 8) attention + x_1 (error cases) 最后，通过 self.gamma 对加权后的特征图进行缩放，
        # 然后与原始的 x_1 进行加权相加。这一步类似于残差连接，允许网络保持原始输入的信息，同时加入了自注意力的加权信息。
        return out, attention # 经过处理后的 out 的大小将是与 x_0 的大小相同，即 (B, C_0, H, W)



@persistence.persistent_class
class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, Fusion=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=1, padding=1, kernel_size=3),
            GroupNorm(out_channels),
            nn.Conv2d(out_channels, out_channels, stride=1, padding=1, kernel_size=3),
            GroupNorm(out_channels),
        )
        self.Fusion = Fusion
        self.self_se_atten = SelfAttention(num_attention_heads=4, input_size=out_channels, hidden_size=out_channels,
                                           hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
        self.channels = out_channels

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = silu(self.conv(x))
        b, c, h, w = x.shape

        x_ = x.reshape(b, c, w * h).permute(0, 2, 1).contiguous()
        x_ = self.self_se_atten(x_)
        x_ = x_.permute(0, 2, 1).view(b, c, h, w)
        x = x + x_
        return x



@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        # print(freqs.size()) # torch.Size([8])
        # print(x.size()) # torch.Size([1])
        x = x.ger(freqs.to(x.dtype))#ger做外积
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch

@persistence.persistent_class
class SongUNet(torch.nn.Module):#对应ddpmpp，这个就是我们要用的模型框架。
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        # model_channels=16
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        # channel_mult=[1, 2, 4, 8, 16, 32]
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 2,            # Number of residual blocks per resolution.
        attn_resolutions    = [],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        # embedding_type='positional'
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        # channel_mult_noise=1
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        # encoder_type='standard'
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        # decoder_type='standard'
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        # resample_filter=[1, 1]
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout # 0
        emb_channels = model_channels * channel_mult_emb # 64
        noise_channels = model_channels * channel_mult_noise # 16
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        ) # emb_channels=64, num_heads=1, dropout=0.1, skip_scale=np.sqrt(0.5), eps=1e-6,resample_filter=[1, 1]

        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        # self.map_noise = PositionalEmbedding(num_channels=16, endpoint=True)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None # 任何值为 零、空、None、False、
        # 空容器（如空列表、空字符串等） 都会被视为 False，而其他任何非零值都会被视为 True。所以就是不会创建self.map_label层。
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None # 不会创建self.map_augment层。
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        # self.map_layer0 = Linear(in_features=16, out_features=64, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        # self.map_layer1 = Linear(in_features=64, out_features=64, **init)

        self.fusion = Fusion(in_channels=1024, out_channels=512) # 原896尺寸时候，train函数层数设置为6层时对应的参数。
        self.cross_attention = Cross_Attn(in_dim=512)
        # self.fusion = Fusion(in_channels=512, out_channels=256)
        # self.cross_attention = Cross_Attn(in_dim=256)

        self.enc = torch.nn.ModuleDict()
        cout = in_channels # 看情况，单切片就是2或3，多切片就是9
        caux = in_channels
        for level, mult in enumerate(channel_mult): # channel_mult=[1, 2, 4, 8, 16, 32]
            res = img_resolution >> level # 例如，假设 img_resolution 的值为 16（即二进制 10000），如果 level = 1，那么 img_resolution >> 1 会得到 8（即二进制 1000）。
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs) # emb_channels=64, num_heads=1, adaptive_scale=False
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks): # num_blocks=2
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions) # []
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]  # 对于每一个符合条件（name 不包含 'aux'）的 block，代码会提取该 block 的 out_channels 属性。
        # out_channels 通常表示该层输出的通道数，常见于卷积层或其他类型的网络层。

        # Decoder.这段代码基本上构建了类似U-Net的架构的解码器部分
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):# 循环遍历了通道数量乘数的反向列表,即[32, 16, 8, 4, 2, 1]
            res = img_resolution >> level
            if level == len(channel_mult) - 1:#在循环中，它检查当前级别并相应地创建解码器块。如果是最后一级，则为相同分辨率创建两个块。第一个块启用了注意力，第二个没有。
#如果不是最后一级，则创建一个上采样块。
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop() # 这意味着 pop() 会移除并返回 skips 中的最后一个元素，并且 skips 列表的大小会减少 1。
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, previous_x, cond_pre, x, x_cond, next_x, cond_ne, noise_labels, class_labels, augment_labels=None):
        x = torch.cat([x, x_cond], dim=1)
        # print(x.size()) # torch.Size([1, 3, 896, 896])
        previous_x = torch.cat([previous_x, cond_pre], dim=1)
        next_x = torch.cat([next_x, cond_ne], dim=1)
        fin_x = torch.cat([previous_x, x, next_x], dim=0)
        emb = self.map_noise(noise_labels) # 使用噪声标签（noise_labels）进行嵌入映射。经过这些操作后，输出的形状是 (batch, 16)即3，16。
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos对嵌入进行形状调整，然后交换其中的正弦和余弦部分。
        if self.map_label is not None: # 不走这里
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None: # 不走这里
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))#对嵌入结果分别通过两个映射层，并使用SiLU激活函数进行激活。
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = fin_x # 初始化辅助变量aux为输入数据x。
        for name, block in self.enc.items():#items()返回构建的ModuleDict()里面可迭代的键值对。
            if 'aux_down' in name:
                aux = block(aux)

            elif 'aux_skip' in name:
                fin_x = skips[-1] = fin_x + block(aux)

            elif 'aux_residual' in name:
                fin_x = skips[-1] = aux = (fin_x + block(aux)) / np.sqrt(2)

            else: # 只走这里。
                fin_x = block(fin_x, emb) if isinstance(block, UNetBlock) else block(fin_x)
                skips.append(fin_x) # 如果是UNetBlock则输入x和emb，如果是conv2d则输入x。
        # print('encoder output', x.size()) # encoder output torch.Size([1, 512, 28, 28])

        # Fusion block 暂时不考虑加额外的归一化和激活函数的操作。
        x_pre, x, x_ne = fin_x[0:1, :, :, :], fin_x[1:2, :, :, :], fin_x[2:3, :, :, :]
        fusion_1 = self.fusion(x, x_ne)
        # print(fusion_1.size()) # torch.Size([1, 512, 28, 28])
        cro_attn_1, _ = self.cross_attention(x_pre, fusion_1)
        fusion_2 = self.fusion(x_pre, x_ne)
        cro_attn_2, _ = self.cross_attention(x, fusion_2)
        fusion_3 = self.fusion(x_pre, x)
        cro_attn_3, _ = self.cross_attention(x_ne, fusion_3)
        # print(cro_attn_3.size()) # torch.Size([1, 512, 28, 28])
        x = torch.cat([cro_attn_1, cro_attn_2, cro_attn_3], dim=0)
        # print(x.size()) # torch.Size([3, 512, 28, 28])


        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name: # 完全不会走这里。
                aux = block(aux)

            elif 'aux_norm' in name: # 会走这里，但是只走一次，在倒数第二层layer。
                tmp = block(x)

            elif 'aux_conv' in name: # 如果块的名称中包含'aux_conv'，则将tmp通过该块进行卷积，并使用SiLU激活函数进行激活，并将结果赋给tmp；同时将tmp加到aux上。
            # 会走这里，但是只走一次，在倒数第一层layer。
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux

            else: # 基本上绝大多数情况都走这里。
                if x.shape[1] != block.in_channels:
                    # shape = x.shape[1]
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        # print('decoder output', aux.size()) # decoder output torch.Size([1, 1, 896, 896])

        return aux

#----------------------------------------------------------------------------
# Reimplementation of the ADM architecture from the paper
# "Diffusion Models Beat GANS on Image Synthesis". Equivalent to the
# original implementation by Dhariwal and Nichol, available at
# https://github.com/openai/guided-diffusion

@persistence.persistent_class
class DhariwalUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels. 最终设置model_channels=16
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.通道数的每分辨率乘数。 最终设置channel_mult=[1, 2, 4, 8, 16, 32]
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.嵌入向量维数的乘数。
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.自注意力的分辨率列表。
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb#乘以4
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(emb_channels=emb_channels, channels_per_head=64, dropout=dropout, init=init, init_zero=init_zero)#channels_per_head为单个头的通道

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0 = Linear(in_features=model_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.map_label = Linear(in_features=label_dim, out_features=emb_channels, bias=False, init_mode='kaiming_normal', init_weight=np.sqrt(label_dim)) if label_dim else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):#0，1； 1，2； 2，3； 3，4
            res = img_resolution >> level#计算当前层级的图像分辨率 res，这里使用了位运算 >> 来对图像分辨率进行降采样。右移动运算符
            if level == 0:#如果当前层级是第一层（即最底层）：将输入通道数赋值给 cin。根据模型的通道数和倍数计算当前层级的输出通道数，并赋值给 cout。
                # 创建一个卷积层，并将其添加到 self.enc 中，用于对输入数据进行卷积处理。
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:#如果当前层级不是第一层（即不是最底层）：创建一个 UNetBlock，将其添加到 self.enc 中，用于下采样数据。
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
            for idx in range(num_blocks):#对于每个层级中的每个块，进行以下操作：将当前输出通道数赋值给 cin。根据模型的通道数和倍数计算当前层级的输出通道数，并赋值给 cout。
#创建一个 UNetBlock，将其添加到 self.enc 中，用于对数据进行处理。
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
        skips = [block.out_channels for block in self.enc.values()]#创建了一个列表 skips，其中包含了编码器中每个子模块的输出通道数。enc.values()能返回具体的网络结构。

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):#0，1； 1，2； 2，3； 3，4倒过来。
            res = img_resolution >> level
            if level == len(channel_mult) - 1:#如果level=3
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)#attention=True即应用注意力机制。
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()#从最后一位开始移除
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=(res in attn_resolutions), **block_kwargs)
                #此处的attention指的是分辨率吗？
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels) #noise_labels即输入到PositionalEmbedding函数里面的x，总之就是做了embedding操作
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels) #增强操作
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)#经过了两层线性层。
        if self.map_label is not None:#执行下面语句。
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)#Dropout，设置条件并部分置零。
            emb = emb + self.map_label(tmp)#加了一个class_labels的线性变换。
        emb = silu(emb)

        # Encoder.
        skips = []
        for block in self.enc.values():#变换以后的noise_labels就是要输入到UNetblock里面的emb了。
            x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            skips.append(x)

        # Decoder.
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
        x = self.out_conv(silu(self.out_norm(x)))
        return x

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        beta_d          = 19.9,         # Extent of the noise level schedule.
        beta_min        = 0.1,          # Initial slope of the noise level schedule.
        M               = 1000,         # Original number of timesteps in the DDPM formulation.
        epsilon_t       = 1e-5,         # Minimum t-value used during training.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VEPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Image resolution.
        img_channels,                   # Number of color channels.
        label_dim       = 0,            # Number of class labels, 0 = unconditional.
        use_fp16        = False,        # Execute the underlying model at FP16 precision?
        sigma_min       = 0.02,         # Minimum supported noise level.
        sigma_max       = 100,          # Maximum supported noise level.
        model_type      = 'SongUNet',   # Class name of the underlying model.
        **model_kwargs,                 # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

#----------------------------------------------------------------------------
# Preconditioning corresponding to improved DDPM (iDDPM) formulation from
# the paper "Improved Denoising Diffusion Probabilistic Models".

@persistence.persistent_class
class iDDPMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        C_1             = 0.001,            # Timestep adjustment at low noise levels.
        C_2             = 0.008,            # Timestep adjustment at high noise levels.
        M               = 1000,             # Original number of timesteps in the DDPM formulation.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=img_channels*2, label_dim=label_dim, **model_kwargs)

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        self.register_buffer('u', u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + 1).sqrt()
        c_noise = self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x[:, :self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1), self.u.reshape(1, -1, 1)).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DhariwalUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels, out_channels=1, label_dim=label_dim, **model_kwargs)


    def forward(self, previous_x, cond_pre, x, x_cond, next_x, cond_ne, sigma, class_labels=None, force_fp32=False, **model_kwargs):

        x = x.to(torch.float32)
        previous_x = previous_x.to(torch.float32)
        next_x = next_x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4 # 四个变量 c_skip、c_out、c_in 和 c_noise 都会保持形状 (3, 1, 1, 1)，与输入的 sigma 张量的形状一致。

        F_x = self.model((c_in[0:1] * previous_x).to(dtype), cond_pre, (c_in[1:2] * x).to(dtype), x_cond, (c_in[2:3] * next_x).to(dtype), cond_ne,
                         c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        # c_noise(3,1,1,1)经过 .flatten() 后的形状：(3)
        assert F_x.dtype == dtype
        total_x = torch.cat([previous_x, x, next_x], dim=0)
        D_x = c_skip * total_x + c_out * F_x.to(torch.float32)

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)#torch.as_tensor这个方法还是比较直观地，将数据转化为tensor


#----------------------------------------------------------------------------

