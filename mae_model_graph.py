import os
import sys
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import SAGEConv,LayerNorm
from mae_utils import get_sinusoid_encoding_table,Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from torch_geometric.nn import MessagePassing
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM



def nll_loss(h, y, c, alpha=0.5, eps=1e-7, reduction='mean'):
    len_c = -c.shape[0]
    h = h[len_c:]
    y = y[len_c:]
    
    # h 是预测的4个标准
    # h 是预测的4个标准
    # y 是真实的标准
    # c 是真实的标准的类别
    # make sure these are ints
    y = torch.tensor(y)
    y = y.type(torch.int64).unsqueeze(1).to(device)
    c = torch.tensor(c)
    c = c.type(torch.int64).unsqueeze(1).to(device)
    s = 1-c
    hazards = torch.sigmoid(h)
    S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1).to(device)
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    uncensored_loss = -(1 - s) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - s * torch.log(s_this)
    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def output_head(dim: int, num_classes: int):
    """
    Creates a head for the output layer of a model.

    Args:
        dim (int): The input dimension of the head.
        num_classes (int): The number of output classes.

    Returns:
        nn.Sequential: The output head module.
    """
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )

class VisionEncoderMambaBlock(nn.Module):
  
    def __init__(
        self,
        dim: int,
        heads: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        ).to(device)
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        ).to(device)
        self.norm = nn.LayerNorm(dim).to(device)
        self.activation = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state).to(device)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim).to(device)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
       
        
        skip = x
       
        
        x = self.norm(x)
        
        
        z1 = self.proj(x)
        
        x1 = self.proj(x)
      
        
        # forward con1d
        x1_rearranged = rearrange(x1, "b s d -> b d s")
        forward_conv_output = self.forward_conv1d(x1_rearranged)
        forward_conv_output = rearrange(
            forward_conv_output, "b d s -> b s d"
        )

        x1_ssm = self.ssm(forward_conv_output)
       
        # backward conv x2
        x2_rearranged = rearrange(x1, "b s d -> b d s")
        
        x2 = self.backward_conv1d(x2_rearranged)
        x2 = rearrange(x2, "b d s -> b s d")
    
        # Backward ssm
        x2 = self.ssm(x2)

        # Activation
        z = self.activation(z1)
        # matmul with z + backward ssm
     
        x2 = x2 * z
       
        # Matmul with z and x1
        x1 = x1_ssm * z
  
        x = x1 + x2
   
        return x + skip

class Vim2(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dt_rank: int = 32,
        dim_inner: int = None,
        d_state: int = None,
        
        dropout: float = 0.1,
        depth: int = 12,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state
        self.dropout = dropout
        self.depth = depth

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Latent
        self.to_latent = nn.Identity()

        # encoder layers
        self.layers = nn.ModuleList()

        # Append the encoder layers
        for _ in range(depth):
            self.layers.append(
                VisionEncoderMambaBlock4(
                    dim=dim,
                    heads=heads,
                    dt_rank=dt_rank,
                    dim_inner=dim_inner,
                    d_state=d_state,
                    *args,
                    **kwargs,
                )
            )

        # Output head
        

    def forward(self, x: Tensor):
        # Patch embedding
        c, h, w = x.shape
      
        # Dropout
        x = self.dropout(x)
        x1 = x[0][0].unsqueeze(0).unsqueeze(0)
        x2 = x[0][1].unsqueeze(0).unsqueeze(0)
        x3 = x[0][2].unsqueeze(0).unsqueeze(0)

        # Forward pass with the layers
        for layer in self.layers:
            
            a,b,c = layer(x1,x2,x3)
        # Latent
        a = self.to_latent(a)
        b = self.to_latent(b)
        c = self.to_latent(c)
        # Output head with the cls tokens
        feature = torch.cat((a, b, c), dim=1)
        return feature

class VisionEncoderMambaBlock3(nn.Module):
  
    def __init__(
        self,
        dim: int,
        heads: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        ).to(device)
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        ).to(device)
        self.norm = nn.LayerNorm(dim).to(device)
        self.activation = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state).to(device)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim).to(device)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x,x_,x__):
              # x is of shape [batch_size, seq_len, dim]
        b, s, d = x.shape
        b_, s_, d_ = x_.shape
        
        skip = x
        skip_ = x_
        skip__ = x__
        
        x = self.norm(x)
        x_ = self.norm(x_)
        x__ = self.norm(x__)
        
        z1 = self.proj(x)
        z1_ = self.proj(x_)
        z1__ = self.proj(x__)
        
        x1 = self.proj(x)
        x1_ = self.proj(x_)
        x1__ = self.proj(x__)
        
        # forward con1d
        x1_rearranged = rearrange(x1, "b s d -> b d s")
        forward_conv_output = self.forward_conv1d(x1_rearranged)
        forward_conv_output = rearrange(
            forward_conv_output, "b d s -> b s d"
        )

        x1_rearranged_ = rearrange(x1_, "b s d -> b d s")
        forward_conv_output_ = self.forward_conv1d(x1_rearranged_)
        forward_conv_output_ = rearrange(
            forward_conv_output_, "b d s -> b s d"
        )

        x1_rearranged__ = rearrange(x1__, "b s d -> b d s")
        forward_conv_output__ = self.forward_conv1d(x1_rearranged__)
        forward_conv_output__ = rearrange(
            forward_conv_output__, "b d s -> b s d"
        )

        x1_ssm = self.ssm(forward_conv_output)
        x1_ssm_ = self.ssm(forward_conv_output_)
        x1_ssm__ = self.ssm(forward_conv_output__)


        
        # Activation
        z = self.activation(z1)
        z_ = self.activation(z1_)
        z__ = self.activation(z1__)





        # matmul with z + backward ssm
        
        # Matmul with z and x1
        a = x1_ssm * z_
        b = x1_ssm * z__
        c = x1_ssm_ * z
        d = x1_ssm_ * z__
        e = x1_ssm__ * z
        f = x1_ssm__ * z
        
        return a+b+c+d+e+f


class VisionEncoderMambaBlock4(nn.Module):
  
    def __init__(
        self,
        dim: int,
        heads: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        ).to(device)
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        ).to(device)
        self.norm = nn.LayerNorm(dim).to(device)
        self.activation = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state).to(device)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim).to(device)

        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x,x_,x__):
              # x is of shape [batch_size, seq_len, dim]
        b, s, d = x.shape
        b_, s_, d_ = x_.shape
        
        skip = x
        skip_ = x_
        skip__ = x__
        
        x = self.norm(x)
        x_ = self.norm(x_)
        x__ = self.norm(x__)
        
        z1 = self.proj(x)
        z1_ = self.proj(x_)
        z1__ = self.proj(x__)
        
        x1 = self.proj(x)
        x1_ = self.proj(x_)
        x1__ = self.proj(x__)
        
        # forward con1d
        x1_rearranged = rearrange(x1, "b s d -> b d s")
        forward_conv_output = self.forward_conv1d(x1_rearranged)
        forward_conv_output = rearrange(
            forward_conv_output, "b d s -> b s d"
        )

        x1_rearranged_ = rearrange(x1_, "b s d -> b d s")
        forward_conv_output_ = self.forward_conv1d(x1_rearranged_)
        forward_conv_output_ = rearrange(
            forward_conv_output_, "b d s -> b s d"
        )

        x1_rearranged__ = rearrange(x1__, "b s d -> b d s")
        forward_conv_output__ = self.forward_conv1d(x1_rearranged__)
        forward_conv_output__ = rearrange(
            forward_conv_output__, "b d s -> b s d"
        )

        x1_ssm = self.ssm(forward_conv_output)
        x1_ssm_ = self.ssm(forward_conv_output_)
        x1_ssm__ = self.ssm(forward_conv_output__)


        # backward con1d

        x2_rearranged = rearrange(x1, "b s d -> b d s")
        x2_rearranged_ = rearrange(x1_, "b s d -> b d s")
        x2_rearranged__ = rearrange(x1__, "b s d -> b d s")

        x2 = self.backward_conv1d(x2_rearranged)
        x2_ = self.backward_conv1d(x2_rearranged_)
        x2__ = self.backward_conv1d(x2_rearranged__)

        x2 = rearrange(x2, "b d s -> b s d")
        x2_ = rearrange(x2_, "b d s -> b s d")
        x2__ = rearrange(x2__, "b d s -> b s d")

        x2 = self.ssm(x2)
        x2_ = self.ssm(x2_)
        x2__ = self.ssm(x2__)

        # Activation
        z = self.activation(z1)
        z_ = self.activation(z1_)
        z__ = self.activation(z1__)
        # matmul with z + backward ssm


        x2 = x2 * z
        x2_ = x2_ * z_
        x2__ = x2__ * z__

        x1 = x1_ssm * z
        x1_ = x1_ssm_ * z_
        x1__ = x1_ssm__ * z__

        x = x1 + x2
        x_ = x1_ + x2_
        x__ = x1__ + x2__

        
        # Matmul with z and x1
        
        
        return x+skip, x_+skip_, x__+skip__


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)


    def forward(self, x, batch, size=None):
        """"""
        
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        
        
        size = batch[-1].item() + 1 if size is None else size
        
        
        gate = self.gate_nn(x).view(-1, 1)
        
        
        x = self.nn(x) if self.nn is not None else x
        
        
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        
        
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)
    
        weighted_x = gate*x
        
        

        return out,gate,weighted_x


    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)
    
    
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    
class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False,train_type_num=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

        self.patch_embed = nn.Linear(embed_dim,embed_dim)
        num_patches = train_type_num

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):

        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()   # sine-cosine positional embeddings

        B, _, C = x.shape
    
        # print('x',x.shape)
        # print('mask',mask.shape)
        # print('mask',mask.shape)
        mask = np.squeeze(mask, axis=0)
        
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible  (1,1,512)
        
        # print('x_vis',x_vis.shape)

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)

        # print('x_vis_final',x_vis.shape)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)  #将未mask的encoder一下
        x = self.head(x)
        
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=512, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,train_type_num=3,
                 ):
        super().__init__()
        self.num_classes = num_classes
#         assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=512, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=512, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.3,
                 drop_path_rate=0.3, 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 train_type_num=3,
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            train_type_num=train_type_num)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=3,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            train_type_num=train_type_num)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        
#         self.mask_token = torch.zeros(1, 1, decoder_embed_dim).to(device)
        

        self.pos_embed = get_sinusoid_encoding_table(train_type_num, decoder_embed_dim)  # 生成不可训练位置embeding矩阵

        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        
        mask = [[[False,False,False]]]


        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]

        # print('x_vis_trsnformer_encoder',x_vis.shape)

        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        # print('x_vis_trsnformer_encoder_to_decoder',x_vis.shape)
        count_true =np.sum(mask)
        # print(count_true)
        self.mask_token = nn.Parameter(torch.zeros(1, count_true, x_vis.shape[2])).to(device)
        trunc_normal_(self.mask_token, std=.02)



        # print('x_vis_trsnformer_decoder',x_vis.shape)
        
        B, N, C = x_vis.shape
        # print('x_vis',x_vis.shape)
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        # print('expand_pos_embed',expand_pos_embed.shape)
        mask1 = np.squeeze(mask, axis=0)

        pos_emd_vis = expand_pos_embed[~mask1].reshape(B, -1, C)
        # print('pos_emd_vis',pos_emd_vis.shape)
        
        pos_emd_mask = expand_pos_embed[mask1].reshape(B, -1, C)
        # print('pos_emd_mask',pos_emd_mask.shape)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        # print('self.mask_token',self.mask_token.shape)
        # print('x_full',x_full.shape)

        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.decoder(x_full, 0) # [B, N_mask, 3 * 16 * 16]

        # print('x__decoder',x.shape)

        tmp_x = torch.zeros_like(x).to(device)
        Mask_n = 0
        Truth_n = 0
        for i,flag in enumerate(mask[0][0]):
            if flag:  
                tmp_x[:,i] = x[:,pos_emd_vis.shape[1]+Mask_n]
                Mask_n += 1
            else:
                tmp_x[:,i] = x[:,Truth_n]
                Truth_n += 1
        # tmp_x = x_vis
        return tmp_x



def Mix_mlp(dim1):
    
    return nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.GELU(),
            nn.Linear(dim1, dim1))

class MixerBlock(nn.Module):
    def __init__(self,dim1,dim2):
        super(MixerBlock,self).__init__() 
        
        self.norm = LayerNorm(dim1)
        self.mix_mip_1 = Mix_mlp(dim1)
        self.mix_mip_2 = Mix_mlp(dim2)
        
    def forward(self,x): 
        
        # print('x____.shape',x.shape)

        # y = self.norm(x)
       

        y = x.transpose(0,1)
        
        y = self.mix_mip_1(y)
        
        y = y.transpose(0,1)
        x = x + y
        # y = self.norm(x)
        x = x + self.mix_mip_2(y)
        
#         y = self.norm(x)
#         y = y.transpose(0,1)
#         y = self.mix_mip_1(y)
#         y = y.transpose(0,1)
#         x = self.norm(y)
        return x


class MixerBlock2(nn.Module):
    
    def __init__(
        self,
        dim: int,
        heads: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        ).to(device)
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        ).to(device)
        self.norm = nn.LayerNorm(dim).to(device)
        self.activation = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state).to(device)

        # Linear layer for z and x
        self.proj = nn.Linear(dim, dim).to(device)

        # Softplus
        self.softplus = nn.Softplus()
        
    def forward(self,x): 
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x1_ = x1.unsqueeze(0).unsqueeze(0)
        x2_ = x2.unsqueeze(0).unsqueeze(0)
        x3_ = x3.unsqueeze(0).unsqueeze(0)
        block = VisionEncoderMambaBlock4(dim=512, heads=1, dt_rank=32,dim_inner=512, d_state=256)
        a,b,c = block(x1_,x2_,x3_)
        a = a.squeeze(0)
        b = b.squeeze(0)
        c = c.squeeze(0)
        a = torch.cat((a,b),0)
        a = torch.cat((a,c),0)
       
        return a



def MLP_Block(dim1, dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout))

def GNN_relu_Block(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
#             GATConv(in_channels=dim1,out_channels=dim2),
            nn.ReLU(),
            LayerNorm(dim2),
            nn.Dropout(p=dropout))






class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, normalize_emb, aggr="mean", **kwargs):
        super(EdgeSAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.normalize_emb = normalize_emb

        self.message_lin = nn.Linear(in_channels + edge_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)
        self.message_activation = nn.ReLU()
        self.update_activation = nn.ReLU()

    def forward(self, x, edge_attr, edge_index):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, edge_index):
        
    
        m_j = torch.cat((x_j, edge_attr), dim=-1)
        m_j = self.message_activation(self.message_lin(m_j))

        return m_j

    def update(self, aggr_out, x):
        aggr_out = self.update_activation(self.agg_lin(torch.cat((aggr_out, x), dim=-1)))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, edge_channels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels,
                                                     self.edge_channels)


class GNNStack(torch.nn.Module):
    def __init__(self, node_channels, edge_channels, normalize_embs, num_layers, dropout):
        super(GNNStack, self).__init__()
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.normalize_embs = normalize_embs
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = self.build_convs(node_channels, edge_channels, normalize_embs, num_layers)
        self.edge_update_mlps = self.build_edge_update_mlps(node_channels, edge_channels, num_layers)

    def build_convs(self, node_channels, edge_channels, normalize_embs, num_layers):
        convs = nn.ModuleList()
        for l in range(num_layers):
            conv = EdgeSAGEConv(node_channels, node_channels, edge_channels, normalize_embs)
            convs.append(conv)
        return convs

    def build_edge_update_mlps(self, node_channels, edge_channels, num_layers):
        edge_update_mlps = nn.ModuleList()
        for l in range(num_layers - 1):
            edge_update_mlp = nn.Sequential(
                nn.Linear(node_channels + node_channels + edge_channels, edge_channels),
                nn.ReLU()
            )
            edge_update_mlps.append(edge_update_mlp)
        return edge_update_mlps

    def update_edge_attr(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]
        edge_attr = mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return edge_attr

    def forward(self, x, edge_attr, edge_index):
        for l, conv in enumerate(self.convs):
            x = conv(x, edge_attr, edge_index)
            if l < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
                edge_attr = self.update_edge_attr(x, edge_attr, edge_index, self.edge_update_mlps[l])
        return x


class fusion_model_mae_2(nn.Module):
    def __init__(self,in_feats,n_hidden,out_classes,dropout=0.3,train_type_num=3):
        super(fusion_model_mae_2,self).__init__() 


        

        self.img_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes)         
        self.img_relu_2 = GNN_relu_Block(out_classes)  
        self.rna_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes)          
        self.rna_relu_2 = GNN_relu_Block(out_classes)      
        self.cli_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes)         
        self.cli_relu_2 = GNN_relu_Block(out_classes) 
#         TransformerConv

        att_net_img = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_img = my_GlobalAttention(att_net_img)

        att_net_rna = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_rna = my_GlobalAttention(att_net_rna)        

        att_net_cli = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_cli = my_GlobalAttention(att_net_cli)


        att_net_img_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_img_2 = my_GlobalAttention(att_net_img_2)

        att_net_rna_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_rna_2 = my_GlobalAttention(att_net_rna_2)        

        att_net_cli_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_cli_2 = my_GlobalAttention(att_net_cli_2)
        
        
        
        self.mae = PretrainVisionTransformer(encoder_embed_dim=out_classes, decoder_num_classes=out_classes, decoder_embed_dim=out_classes, encoder_depth=1,decoder_depth=1,train_type_num=train_type_num)
        self.mamba =  Vim2(
        dim=512,  # Dimension of the model
        heads=16,  # Number of attention heads
        dt_rank=32,  # Rank of the dynamic routing tensor
        dim_inner=512,  # Inner dimension of the model
        d_state=256,  # State dimension of the model
        dropout=0,  # Dropout rate
        depth=2,  # Depth of the model
        )
        
        # self.mix = MixerBlock2(dim=512, heads=16, dt_rank=32,dim_inner=512, d_state=256)
        self.mix = MixerBlock(train_type_num, out_classes)
        
        self.lin1_img = torch.nn.Linear(out_classes,out_classes//4)
        self.lin1_img_a = torch.nn.Linear(out_classes//4,4)
        self.lin2_img = torch.nn.Linear(out_classes//4,1)   
        self.lin2_img_four = torch.nn.Linear(out_classes//4,4)      
        self.lin1_rna = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_rna = torch.nn.Linear(out_classes//4,1)
        self.lin2_rna_four = torch.nn.Linear(out_classes//4,4) 
        self.lin1_cli = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_cli = torch.nn.Linear(out_classes//4,1)
        self.lin2_cli_four = torch.nn.Linear(out_classes//4,4)
                 

        self.norm_img = LayerNorm(out_classes//4)
        self.norm_rna = LayerNorm(out_classes//4)
        self.norm_cli = LayerNorm(out_classes//4)
        self.norm_other = LayerNorm(out_classes)
        self.relu = torch.nn.ReLU() 


        self.cl_projection = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
        )
        self.modality_nodes = nn.Parameter(torch.randn(3, n_hidden))
        normalize_embs = [True] * 2
        self.gnn = GNNStack(512, 512, normalize_embs, 2, dropout)
        self.tau = nn.Parameter(torch.tensor(1 / 0.07), requires_grad=False)
        self.dropout=nn.Dropout(p=dropout)
        self.risk1 = torch .nn.Linear(512,128)
        self.risk2 = torch.nn.Linear(128,1)
    
    def edgedrop(self, flag):
        n, m = flag.size()
        for i in range(n):
            count_ones = flag[i].sum().item()

            if torch.rand(1) < 0.5:
                continue

            if count_ones <= 1:
                continue

            # Randomly choose how many 1s we want to keep
            keep_count = random.randint(1, count_ones)
    
            # Get the indices of 1s in the row
            one_indices = (flag[i] == 1).nonzero(as_tuple=True)[0]

            # Randomly shuffle the indices
            one_indices = one_indices[torch.randperm(one_indices.size(0))]
            
            # Set the 1s we don't want to keep to 0
            mask_count = count_ones - keep_count
            
            flag[i][one_indices[:int(mask_count)]] = 0

        return flag
    
    def unsup_ce_loss(self, zaz_s):
        target = torch.arange(zaz_s.size(0), device=zaz_s.device)
        loss = F.cross_entropy(zaz_s, target)
        loss_t = F.cross_entropy(zaz_s.t(), target)
        return (loss + loss_t) / 2


    def forward(self,all_thing,train_use_type=None,use_type=None,in_mask=[],mix=False):

        block = VisionEncoderMambaBlock(dim=512, heads=16, dt_rank=32, dim_inner=512, d_state=256)
        if len(in_mask) == 0:
        
            mask = np.array([[[False]*len(train_use_type)]])
        else:
            mask = in_mask
        

        data_type = use_type
        x_img = all_thing.x_img
        x_rna = all_thing.x_rna
        x_cli = all_thing.x_cli

        data_id=all_thing.data_id
        edge_index_img=all_thing.edge_index_image
        edge_index_rna=all_thing.edge_index_rna
        edge_index_cli=all_thing.edge_index_cli

        
        save_fea = {}
        fea_dict = {}
        num_img = len(x_img)
        num_rna = len(x_rna)
        num_cli = len(x_cli)      
               
            
        att_2 = []
        att_each = []
        pool_x = torch.empty((0)).to(device)
        if 'img' in data_type:
            
            x_img = self.img_gnn_2(x_img,edge_index_img) 
            x_img = self.img_relu_2(x_img)  #n*512  
            batch = torch.zeros(len(x_img),dtype=torch.long).to(device)
            pool_x_img,att_img_2,att_img_each = self.mpool_img(x_img,batch)# pool_x_img 1*512
            # print('pool_x_img',pool_x_img.shape)
            att_2.append(att_img_2)
            att_each.append(att_img_each)
            pool_x = torch.cat((pool_x,pool_x_img),0)
        if 'rna' in data_type:
            print('rna',x_rna.shape)
            x_rna = self.rna_gnn_2(x_rna,edge_index_rna) 
            print('x_rna',x_rna.shape)
            x_rna = self.rna_relu_2(x_rna)   
            batch = torch.zeros(len(x_rna),dtype=torch.long).to(device)
            pool_x_rna,att_rna_2,att_rna_each = self.mpool_rna(x_rna,batch)
            att_2.append(att_rna_2)
            att_each.append(att_rna_each)
            pool_x = torch.cat((pool_x,pool_x_rna),0)
        if 'cli' in data_type:
            x_cli = self.cli_gnn_2(x_cli,edge_index_cli) 
            x_cli = self.cli_relu_2(x_cli)   
            batch = torch.zeros(len(x_cli),dtype=torch.long).to(device)
            pool_x_cli,att_cli_2,att_cli_each = self.mpool_cli(x_cli,batch)
            att_2.append(att_cli_2)
            att_each.append(att_cli_each)
            pool_x = torch.cat((pool_x,pool_x_cli),0)
        fea_dict['mae_labels'] = pool_x


        if len(train_use_type)>1:     # img','rna','cli
            if use_type == train_use_type:
                
                # print('pool_x',pool_x.unsqueeze(dim=0).shape)
                mae_x  = self.mamba(pool_x.unsqueeze(dim=0)).squeeze(0)
                # mae_x = self.mae(pool_x,mask)
                # print('mae_x',mae_x.shape)
                fea_dict['mae_out'] = mae_x
            else:
                k=0
                tmp_x = torch.zeros((len(train_use_type),pool_x.size(1))).to(device)
                mask = np.ones(len(train_use_type),dtype=bool)
                for i,type_ in enumerate(train_use_type):
                    if type_ in data_type:
                        tmp_x[i] = pool_x[k]
                        k+=1
                        mask[i] = False
                mask = np.expand_dims(mask,0)
                mask = np.expand_dims(mask,0)
                if k==0:
                    mask = np.array([[[False]*len(train_use_type)]])
                
                # print('tmp_x',tmp_x.shape)
                mae_x = self.mamba(tmp_x,mask).squeeze(0)
                # print('mae_x',mae_x.shape)
               
                
                fea_dict['mae_out'] = mae_x   


            save_fea['after_mae'] = mae_x.cpu().detach().numpy() 
            if mix:
                mae_x = self.mix(mae_x)
                save_fea['after_mix'] = mae_x.cpu().detach().numpy() 
                # print('mae_x',mae_x.shape)
            k=0
            x_img_return= x_img
            x_rna_return= x_rna
            x_cli_return= x_cli
            if 'img' in train_use_type and 'img' in use_type:
                x_img = x_img + mae_x[train_use_type.index('img')] 
                k+=1
            if 'rna' in train_use_type and 'rna' in use_type:
                x_rna = x_rna + mae_x[train_use_type.index('rna')]  
                k+=1
            if 'cli' in train_use_type and 'cli' in use_type:
                x_cli = x_cli + mae_x[train_use_type.index('cli')]  
                k+=1
                # print('x_img',x_img.shape)
            
 
        att_3 = []
        pool_x = torch.empty((0)).to(device)

        
        if 'img' in data_type:
            batch = torch.zeros(len(x_img),dtype=torch.long).to(device)
            pool_x_img,att_img_3,_ = self.mpool_img_2(x_img,batch)
            att_3.append(att_img_3)
            pool_x = torch.cat((pool_x,pool_x_img),0)
        if 'rna' in data_type:
            batch = torch.zeros(len(x_rna),dtype=torch.long).to(device)
            pool_x_rna,att_rna_3,_ = self.mpool_rna_2(x_rna,batch)
            att_3.append(att_rna_3)
            pool_x = torch.cat((pool_x,pool_x_rna),0)
        if 'cli' in data_type:
            batch = torch.zeros(len(x_cli),dtype=torch.long).to(device)
            pool_x_cli,att_cli_3,_ = self.mpool_cli_2(x_cli,batch)
            att_3.append(att_cli_3)
            pool_x = torch.cat((pool_x,pool_x_cli),0) 
        x = pool_x
        
        x = F.normalize(x, dim=1)
        fea = x
        
        k=0
        if 'img' in data_type:
            fea_dict['img'] = fea[k]
            k+=1
        if 'rna' in data_type:
            fea_dict['rna'] = fea[k]       
            k+=1
        if 'cli' in data_type:
            fea_dict['cli'] = fea[k]
            k+=1
        # print('fea_dict',fea_dict['cli'].shape)
        
        k=0
        multi_x = torch.empty((0)).to(device)
       
        
        if 'img' in data_type:
            x_img = self.lin1_img(x[k])
            x_img = self.relu(x_img)
            x_img = self.norm_img(x_img)
            x_img_ = self.dropout(x_img)    

            x_img = self.lin2_img(x_img_).unsqueeze(0)
            img_feature = self.lin2_img_four(x_img_).unsqueeze(0)
            multi_x = torch.cat((multi_x,x_img),0)
            k+=1
        if 'rna' in data_type:
            x_rna = self.lin1_rna(x[k])
            x_rna = self.relu(x_rna)
            x_rna = self.norm_rna(x_rna)
            x_rna_ = self.dropout(x_rna) 

            x_rna = self.lin2_rna(x_rna_).unsqueeze(0)
            rna_feature = self.lin2_rna_four(x_rna_).unsqueeze(0)
            multi_x = torch.cat((multi_x,x_rna),0)  
            k+=1
        if 'cli' in data_type:
            x_cli = self.lin1_cli(x[k])
            x_cli = self.relu(x_cli)
            x_cli = self.norm_cli(x_cli)
            x_cli_ = self.dropout(x_cli)

            x_cli = self.lin2_rna(x_cli_).unsqueeze(0) 
            cli_feature = self.lin2_cli_four(x_cli_).unsqueeze(0) 
            multi_x = torch.cat((multi_x,x_cli),0)  
            k+=1  
        one_x = torch.mean(multi_x,dim=0)

        
   
        #return (img_feature,rna_feature,cli_feature), (one_x,multi_x),save_fea,(att_2,att_3),fea_dict  
        return  (one_x,multi_x),save_fea,(att_2,att_3,att_each),fea_dict,(x_img_return,x_rna_return,x_cli_return)  
    
    def graph(self,x1, x1_flag,x2, x2_flag,x3, x3_flag,status_all,x_img_feature_all,x_rna_feature_all,x_cli_feature_all):
        criterion = nn.CrossEntropyLoss() 
        batch_size = x1.shape[0]
        hidden_dim = x1.shape[1]
        mamba_out = torch.empty((0)).to(device)

        block = VisionEncoderMambaBlock3(dim=512, heads=16, dt_rank=16,dim_inner=512, d_state=16)
        
        for i in range(batch_size):
            x1_ = x1[i].unsqueeze(0).unsqueeze(0)
            x2_ = x2[i].unsqueeze(0).unsqueeze(0)
            x3_ = x3[i].unsqueeze(0).unsqueeze(0)
            out = block(x1_,x2_,x3_)
            out = out.squeeze(0)
            mamba_out = torch.cat((mamba_out,out),0)

        x_flag = torch.stack([x1_flag, x2_flag, x3_flag], dim=1)
        x = torch.stack([x1, x2, x3], dim=1)

        # g_patient_nodes = torch.ones(batch_size, hidden_dim).to(device)
    
        g_patient_nodes = (x1+x2+x3)/3
        g_patient_nodes_x_feature = (x_img_feature_all+x_rna_feature_all+x_cli_feature_all)/3

        g_nodes = torch.cat([g_patient_nodes, self.modality_nodes], dim=0)
        g_edge_index = x_flag.nonzero().t()
        g_edge_index[1] += batch_size
        g_edge_index = torch.cat([g_edge_index, g_edge_index.flip([0])], dim=1)
        g_edge_attr = x[x_flag.bool()].repeat(2, 1)
        # print('g_edge_index',g_edge_attr[0])
        # print('x1',x1[0])
        z = self.gnn(g_nodes, g_edge_attr, g_edge_index) 
        
        ag_x_flag = x_flag.clone()
        
        ag_x_flag = self.edgedrop(ag_x_flag)
        ag_x = x*ag_x_flag.unsqueeze(2).float()
        ag_patient_nodes = g_patient_nodes.clone()

        for i in range(ag_x_flag.shape[0]):
            count = ag_x_flag[i].sum().item()
            ag_patient_nodes[i] =  (ag_x[i][0]+ ag_x[i][1]+ ag_x[i][2])/count


        
        
        ag_patient_nodes = ag_patient_nodes.to(x1.device)
        ag_nodes = torch.cat([ag_patient_nodes, self.modality_nodes], dim=0)
        ag_edge_index = ag_x_flag.nonzero().t()
        ag_edge_index[1] += batch_size
        ag_edge_index = torch.cat([ag_edge_index, ag_edge_index.flip([0])], dim=1)
        ag_edge_attr = x[ag_x_flag.bool()].repeat(2, 1)
        az = self.gnn(ag_nodes, ag_edge_attr, ag_edge_index)






        z = z[:batch_size]
        z = z + g_patient_nodes_x_feature
        az = az[:batch_size]
        az = az



        u = self.cl_projection(z)
        u = F.normalize(u, dim=-1)
        au = self.cl_projection(az)
        au = F.normalize(au, dim=-1)
        uau_s = torch.matmul(u, au.t()) * self.tau
        unsup_loss = self.unsup_ce_loss(uau_s)





        # z = self.cl_projection(z)
        z_risk = self.lin1_img(z)
        z_risk = self.relu(z_risk)
        z_risk = self.norm_img(z_risk)
        z_risk = self.dropout(z_risk)
        z_risk = self.lin2_img(z_risk).unsqueeze(0)

        z_risk_ = z_risk.squeeze(0,2)
        riskss = z_risk_.cpu().detach().numpy()
        
        
        labels = [0, 1, 2, 3]
        quantile_bins = pd.qcut(riskss*-1, q=4, labels=labels) 
        quantile_bins = torch.tensor(quantile_bins).to(device)

    
        za_risk =  F.relu(self.lin1_img(az))
        za_risk = self.lin1_img_a(za_risk) 
        za_risk = torch.sigmoid(za_risk)
        
        status_all = torch.tensor(status_all).to(device)
        
        cs_loss = nll_loss(za_risk, quantile_bins, status_all, alpha=0.5, eps=1e-7, reduction='mean')


        return z_risk,unsup_loss,cs_loss
    

    def inference(self, x1, x1_flag, x2, x2_flag, x3, x3_flag,x_img_feature_val,x_rna_feature_val,x_cli_feature_val):

        batch_size = x1.size(0)
        hidden_dim = x1.size(1)

        block = VisionEncoderMambaBlock3(dim=512, heads=16, dt_rank=16,dim_inner=512, d_state=16)
        mamba_out = torch.empty((0)).to(device)

        for i in range(batch_size):
            x1_ = x1[i].unsqueeze(0).unsqueeze(0)
            x2_ = x2[i].unsqueeze(0).unsqueeze(0)
            x3_ = x3[i].unsqueeze(0).unsqueeze(0)
            out = block(x1_,x2_,x3_)
            out = out.squeeze(0)
            mamba_out = torch.cat((mamba_out,out),0)
        

        x_flag = torch.stack([x1_flag, x2_flag, x3_flag], dim=1)
        x = torch.stack([x1, x2, x3], dim=1)

        # g_patient_nodes = torch.ones(batch_size, hidden_dim)
        
        g_patient_nodes = (x1+x2+x3)/3
        g_patient_nodes_x_feature = (x_img_feature_val+x_rna_feature_val+x_cli_feature_val)/3




        g_patient_nodes = g_patient_nodes.to(x1.device)
        g_nodes = torch.cat([g_patient_nodes, self.modality_nodes], dim=0)
        g_edge_index = x_flag.nonzero().t()
        g_edge_index[1] += batch_size
        g_edge_index = torch.cat([g_edge_index, g_edge_index.flip([0])], dim=1)
        g_edge_attr = x[x_flag.bool()].repeat(2, 1)

        z = self.gnn(g_nodes, g_edge_attr, g_edge_index)
        z = z[:batch_size]
        z = z+g_patient_nodes_x_feature

        z_risk = self.lin1_img(z)
        z_risk = self.relu(z_risk)
        z_risk = self.norm_img(z_risk)
        z_risk = self.dropout(z_risk)
        z_risk = self.lin2_img(z_risk).unsqueeze(0)

        return z_risk







