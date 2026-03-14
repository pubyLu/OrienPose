import torch.nn as nn
import torch
import pickle
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1
class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class OrienCrossAttnBlock(nn.Module):
    def __init__(self,
                 c=4, bins=786, num_tokens=16, d_model=None, nhead=1, mlp_ratio=2.0, dropout=0.0):
        super(OrienCrossAttnBlock, self).__init__()
        d_model = d_model or c
        self.dim = d_model // nhead
        self.nhead = nhead
        self.keyEncoder = nn.Sequential(
            nn.Linear(720, bins, bias=False)
        )
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False) # 特征作query
        self.k_proj = nn.Linear(bins, num_tokens*d_model, bias=False)
        self.v_proj = nn.Linear(bins, num_tokens*d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2,3, 1)
        x = x.view(B, H*W, C)
        x = self.norm1(x)
        bs = x.size(0)
        source = self.keyEncoder(source)
        query, key, value = x, source, source
        
        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        # print("query.shape:", query.shape)
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        # print("key.shape:", key.shape)
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)
        y = x + message
        y = y.view(B, H, W, C).permute(0, 3, 1, 2)
        return y