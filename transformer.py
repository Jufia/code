"""
NSA 和 naive transformer 的计算复杂度对比。
"""

import torch
from torch import nn

from models.transformerlayers.layer_norm import LayerNorm
# from models.transformerlayers.multi_head_attention import MultiHeadAttention
from chapter3.naive import nsa as MultiHeadAttention
from models.transformerlayers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, dk=d_model, dv=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):

    def __init__(self, in_channel, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = nn.Linear(in_channel, d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x

class Transformer(nn.Module):
    def __init__(self, in_channel, classn, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               in_channel=in_channel,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.classfier = nn.Sequential(
            nn.Linear(max_len*d_model, max_len),
            nn.ReLU(),
            nn.Linear(max_len, classn),
            nn.Softmax(dim=-1)
        )

    def forward(self, src):
        b, l, _ = src.shape
        enc_src = self.encoder(src, src_mask=None)
        out = self.classfier(enc_src.flatten(1))
        return out
    
if __name__ == '__main__':
    max_len = 1024
    model = Transformer(
        in_channel=6,
        classn=10,
        n_head=8,
        d_model=32,
        max_len=max_len,
        ffn_hidden=64,
        drop_prob=0.1,
        n_layers=6,
        device='cpu'
    )
    x = torch.rand(1, max_len, 6)
    from statestic import *
    print(f'{'*'*20}lenght = {max_len}{'*'*20}')
    print_trainable_parameters(model)
    quality(model, x)
    # y = model(x)
    # print(y.shape)