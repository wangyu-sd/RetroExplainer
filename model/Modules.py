# !/usr/bin/python3
# @File: mol_transformer_2d.py
# --coding:utf-8--
# @Author:yuwang
# @Email:as1003208735@foxmail.com
# @Time: 2022.03.19.13
import copy
from typing import Optional

from torch import nn, Tensor
import torch.nn.functional as F
import math
import torch
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class RetroAGTEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(RetroAGTEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, attn_mask=None):
        output = x
        for mod in self.layers:
            output = mod(output, attn_mask)

        if self.norm is not None and self.num_layers > 0:
            output = self.norm(output)

        return output


class RetroAGTEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False) -> None:
        super(RetroAGTEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAtomAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(RetroAGTEncoderLayer, self).__setstate__(state)

    def forward(self, x, attn_mask=None):

        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class RetroAGTDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(RetroAGTDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, x, attn_mask=None):
        output, output_atten_map = x, None
        for mod in self.layers:
            output, output_atten_map = mod(output, attn_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, output_atten_map


class RetroAGTDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.gelu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False) -> None:
        super(RetroAGTDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAtomAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiHeadAtomAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super(RetroAGTDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        if self.norm_first:
            x += self._sa_block(self.norm1(x), tgt_mask)
            attn_output, attn_output_weights = \
                self._mha_block(self.norm2(x), memory, memory_mask)
            x += attn_output
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            attn_output, attn_output_weights = \
                self._mha_block(x, memory, memory_mask)
            x = self.norm2(x + attn_output)
            x = self.norm2(x + self._ff_block(x))

        return x, attn_output_weights

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, mem, attn_mask):
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                need_weights=False)
        return self.dropout2(x[0]), x[1]

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class MultiHeadAtomAdj(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, add_zero_attn=False,
                 batch_first=False) -> None:
        super(MultiHeadAtomAdj, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((2 * embed_dim, embed_dim)))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(2 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)

    def forward(self, query, key, attn_mask=None):
        if self.batch_first:
            query, key = [x.transpose(1, 0) for x in (query, key)]
        q, k = _in_projection_packed(query, key, None, self.in_proj_weight, self.in_proj_bias)
        q_d, bsz, _ = q.size()
        k_d = k.size(0)
        q = q.contiguous().view(q_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # if attn_mask is not None and attn_mask.size(0) != bsz * self.num_heads:
        #     attn_mask = attn_mask.reshape(bsz, 1, q_d, k_d) \
        #         .expand(bsz, self.num_heads, q_d, k_d).reshape(bsz * self.num_heads, q_d, k_d)

        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)

        _, adj = _scaled_dot_product_atom_attention(q, k, None, 0.0, attn_mask)

        return adj.view(bsz, self.num_heads, q_d, k_d).permute(0, 2, 3, 1).contiguous()


class MultiHeadAtomAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_zero_attn=False,
                 batch_first=False) -> None:
        super(MultiHeadAtomAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias)

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        q_d, bsz, _ = q.size()
        k_d = k.size(0)
        q = q.contiguous().view(q_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(k_d, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # if attn_mask is not None and attn_mask.size(0) != bsz * self.num_heads:
        #     attn_mask = attn_mask.reshape(bsz, 1, q_d, k_d) \
        #         .expand(bsz, self.num_heads, q_d, k_d).reshape(bsz * self.num_heads, q_d, k_d)

        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)

        dropout_p = self.dropout if self.training else 0.0
        attn_output, attn_output_weights = \
            _scaled_dot_product_atom_attention(q, k, v, dropout_p, attn_mask)

        attn_output = attn_output.transpose(0, 1).contiguous().view(q_d, bsz, self.embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if self.batch_first:
            attn_output = attn_output.transpose(1, 0)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, q_d, k_d)
            return attn_output, attn_output_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attn_output, None


def _scaled_dot_product_atom_attention(q, k, v=None, dropout_p=0.0, attn_mask=None):
    """
    :param attn_mask:
    :param q: [bsz, q, d]
    :param k: [bsz, k, d]
    :param v: [bsz, k, d]
    :param dropout_p: p in [0, 1]
    :return:([bsz, q, d], [bsz, q, k]) or (None, [bsz, q, k]) if v is None
    """
    B, Q, D = q.size()
    # print("q.size:", q.size(), k.size())
    q = q / math.sqrt(D)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
        # attn = torch.nan_to_num(attn)
    if v is not None:
        attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    output = torch.bmm(attn, v) if v is not None else None
    # print(f"q:{q}\n"
    #       f"k:{k}\n"
    #       f"v:{v}\n"
    #       f"attn_mask:{attn_mask}"
    #       f"attn:{attn}")
    # raise ValueError
    return output, attn


def _in_projection_packed(q, k, v=None, w=None, b=None):
    """
    :param q: [q, bsz, d]
    :param k: [k, bsz, d]
    :param v: [v, bsz, d]
    :param w: [d*3, d]
    :param b: [d*3]
    :return: projected [q, k, v] or [q, k] if v is None
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return F.linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
    elif v is not None:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    elif q is k:
        return F.linear(q, w, b).chunk(2, dim=-1)
    else:
        w_q, w_k = w.split([E, E])
        if b is None:
            b_q = b_k = None
        else:
            b_q, b_k = b.split([E, E])
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k)


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
