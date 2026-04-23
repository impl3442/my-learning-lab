"""
nanoGPT — 精簡版 GPT-2
架構完全對應 GPT-2 論文，方便之後加模塊改架構。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────

@dataclass
class GPTConfig:
    block_size: int = 1024      # 最大 context 長度
    vocab_size: int = 50257     # GPT-2 預設詞彙表大小
    n_layer:    int = 12        # Transformer 層數
    n_head:     int = 12        # attention head 數
    n_embd:     int = 768       # embedding 維度
    dropout:    float = 0.1


# ─────────────────────────────────────────
# 基本元件
# ─────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    多頭因果自注意力（只看左側 token，不看未來）
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout

        # Q, K, V 一次投影
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 輸出投影
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # 因果遮罩（下三角矩陣）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        head_dim = C // self.n_head

        # 計算 Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        att = (q @ k.transpose(-2, -1)) * scale                  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                               # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)         # 重組 heads

        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """
    Feed-forward 子層（4x 寬度，GELU 激活）
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    """
    一個 Transformer Block：
    LayerNorm → Attention → 殘差
    LayerNorm → MLP      → 殘差
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ─────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte':  nn.Embedding(config.vocab_size, config.n_embd),   # token embedding
            'wpe':  nn.Embedding(config.block_size, config.n_embd),   # position embedding
            'drop': nn.Dropout(config.dropout),
            'h':    nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd),                      # 最後的 LayerNorm
        })

        # 輸出頭（與 token embedding 共享權重）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer['wte'].weight = self.lm_head.weight

        # 初始化權重
        self.apply(self._init_weights)

        print(f"參數量：{self.num_params():,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"序列長度 {T} 超過 block_size {self.config.block_size}"

        # Token + Position Embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer['wte'](idx)   # (B, T, n_embd)
        pos_emb = self.transformer['wpe'](pos)   # (T, n_embd)
        x = self.transformer['drop'](tok_emb + pos_emb)

        # 通過所有 Transformer Block
        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)

        if targets is not None:
            # 訓練時計算 cross-entropy loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            # 推理時只算最後一個 token 的 logits（效率優化）
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回歸生成
        idx: (B, T) 的初始 token 序列
        """
        for _ in range(max_new_tokens):
            # 超過 block_size 就截斷
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ─────────────────────────────────────────
# 快速測試
# ─────────────────────────────────────────

if __name__ == "__main__":
    # 小型設定，方便本地測試
    config = GPTConfig(
        block_size=128,
        vocab_size=50257,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
    )

    model = GPT(config)

    # 隨機輸入
    x = torch.randint(0, config.vocab_size, (2, 32))  # batch=2, seq_len=32
    logits, _ = model(x)
    print(f"輸出 shape：{logits.shape}")  # 應為 (2, 1, 50257)

    # 生成測試
    prompt = torch.zeros((1, 1), dtype=torch.long)
    output = model.generate(prompt, max_new_tokens=10, top_k=40)
    print(f"生成 shape：{output.shape}")  # 應為 (1, 11)