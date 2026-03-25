from turtle import forward
from numpy import block
import torch
from torch._prims_common import set_correction
from torch import nn
import torch.nn.functional as F
import math


def compression(
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int
    ) -> torch.Tensor:
    # Currently, we set mean pooling as our basic compression function.
    B, H, T = k.shape[:3]
    num_block = math.ceil(T / block_size)
    if k.shape[1] % block_size != 0:
        k = F.pad(k, (0, 0, 0, 0, 0, num_block * block_size - T))
        v = F.pad(v, (0, 0, 0, 0, 0, num_block * block_size - T))
    k_cmp = k.view(B, H, num_block, block_size, -1).mean(dim=3)
    v_cmp = v.view(B, H, num_block, block_size, -1).mean(dim=3)
    return k_cmp, v_cmp

class BlockwiseAttention(nn.Module):
    def __init__(self):
        super(BlockwiseAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, BS):
        k_cmp, v_cmp = compression(k, v, BS)

        b, h, T, dk = k.shape
        C = k_cmp.shape[2]
        k_t = k_cmp.transpose(2, 3)  # transpose
        score = q @ k_t / math.sqrt(dk)
        score = self.softmax(score)

        v = score @ v_cmp

        return v, score

class BlockinnerAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, q, k, v, BS, top2_idx):
        b, h, T, dk = k.shape
        num_blocks = T // BS
        k_blocks = k.view(b, h, num_blocks, BS, dk)
        v_blocks = v.view(b, h, num_blocks, BS, dk)

        T_q = top2_idx.shape[2]
        flat_idx = top2_idx.reshape(b, h, T_q * 2)
        flat_idx_exp = flat_idx.unsqueeze(-1).unsqueeze(-1).expand(b, h, T_q * 2, BS, dk)

        k_sel = k_blocks.gather(2, flat_idx_exp).reshape(b, h, T_q, 2 * BS, dk)
        v_sel = v_blocks.gather(2, flat_idx_exp).reshape(b, h, T_q, 2 * BS, dk)

        # q: [b, h, T_q, dk] -> [b, h, T_q, 1, dk]
        # k_sel^T: [b, h, T_q, dk, 2*BS]
        # score: [b, h, T_q, 1, 2*BS] -> squeeze -> [b, h, T_q, 2*BS]
        score = (q.unsqueeze(-2) @ k_sel.transpose(-1, -2)).squeeze(-2) / math.sqrt(dk)
        score = F.softmax(score, dim=-1)

        # score: [b, h, T_q, 1, 2*BS], v_sel: [b, h, T_q, 2*BS, dk]
        out = (score.unsqueeze(-2) @ v_sel).squeeze(-2)  # [b, h, T_q, dk]

        return out

class nsa(nn.Module):
    def __init__(self, d_model, dk, dv, n_head):
        super(nsa, self).__init__()
        self.dq, self.dk, self.dv,self.n_head = dk, dk, dv, n_head

        self.Blockwise = BlockwiseAttention()
        self.Blockinner = BlockinnerAttention()

        self.w_q = nn.Linear(d_model, dk*n_head)
        self.w_k = nn.Linear(d_model, dk*n_head)
        self.w_v = nn.Linear(d_model, dv*n_head)
        self.w_concat = nn.Linear(dv*n_head, d_model)

        self.g_wise = nn.parameter.Parameter(torch.ones(1))
        self.g_inner = nn.parameter.Parameter(torch.ones(1))

    def forward(self, q, k, v, BS=64, mask=None):
        b, max_len, d = q.shape

        qs = self.w_q(q).view(b, max_len, self.n_head, self.dq).transpose(1, 2)
        ks = self.w_k(k).view(b, max_len, self.n_head, self.dk).transpose(1, 2)
        vs = self.w_v(v).view(b, max_len, self.n_head, self.dv).transpose(1, 2)

        # 3. do scale dot product to compute similarity
        wiseatt, score = self.Blockwise(qs, ks, vs, BS)
        top2_idx = score.topk(2, dim=-1).indices
        ineratt = self.Blockinner(qs, ks, vs, BS, top2_idx)

        out = self.g_wise * wiseatt + self.g_inner * ineratt

        # 4. concat and pass to linear layer
        out = out.transpose(1, 2).contiguous().view(b, max_len, self.n_head * self.dv)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

if __name__ == '__main__':
    m = nsa(d_model=32, dk=32, dv=32, n_head=8)
    x = torch.rand(1, 1024, 32)
    y = m(x, x, x, BS=64)
    print(y.shape)