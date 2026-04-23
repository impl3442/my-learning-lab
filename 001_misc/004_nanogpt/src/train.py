"""
train.py — nanoGPT 訓練迴圈
資料集：TinyShakespeare
用法：python train.py
"""

import os
import time
import math
import requests
import numpy as np
import torch
from nanogpt import GPT, GPTConfig


# ─────────────────────────────────────────
# 超參數
# ─────────────────────────────────────────

# 模型
block_size = 256
n_layer    = 6
n_head     = 6
n_embd     = 384
dropout    = 0.1

# 訓練
batch_size    = 32
max_iters     = 5000
eval_interval = 250
eval_iters    = 50
log_interval  = 50

# 優化器
learning_rate = 3e-4
weight_decay  = 0.1
beta1, beta2  = 0.9, 0.95

# 學習率 cosine decay
lr_decay      = True
warmup_iters  = 100
min_lr        = 1e-5

# 儲存
out_dir    = "checkpoints"
checkpoint = "ckpt.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")


# ─────────────────────────────────────────
# 資料準備
# ─────────────────────────────────────────

def download_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = "shakespeare.txt"
    if not os.path.exists(path):
        print("下載 TinyShakespeare...")
        r = requests.get(url)
        with open(path, "w") as f:
            f.write(r.text)
        print("完成")
    return open(path, "r").read()

def prepare_data(text):
    # 字元級 tokenizer（簡單，不需要額外套件）
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, encode, decode

text = download_shakespeare()
train_data, val_data, vocab_size, encode, decode = prepare_data(text)
print(f"詞彙量：{vocab_size}，訓練 tokens：{len(train_data):,}，驗證 tokens：{len(val_data):,}")


# ─────────────────────────────────────────
# 批次取樣
# ─────────────────────────────────────────

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# ─────────────────────────────────────────
# 評估
# ─────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ─────────────────────────────────────────
# 學習率排程（cosine decay with warmup）
# ─────────────────────────────────────────

def get_lr(it):
    if not lr_decay:
        return learning_rate
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    progress = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (learning_rate - min_lr)


# ─────────────────────────────────────────
# 模型初始化
# ─────────────────────────────────────────

config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
)
model = GPT(config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)

os.makedirs(out_dir, exist_ok=True)


# ─────────────────────────────────────────
# 訓練迴圈
# ─────────────────────────────────────────

print("\n開始訓練...\n")
best_val_loss = float('inf')
t0 = time.time()

for it in range(max_iters):

    # 更新學習率
    lr = get_lr(it)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 定期評估
    if it % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | lr {lr:.2e}")

        # 儲存最佳 checkpoint
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save({
                'iter': it,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'val_loss': best_val_loss,
            }, os.path.join(out_dir, checkpoint))

    # 前向 + 反向
    x, y = get_batch("train")
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
    optimizer.step()

    # 定期 log
    if it % log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"  iter {it:5d} | loss {loss.item():.4f} | {dt*1000/log_interval:.1f}ms/iter")


# ─────────────────────────────────────────
# 訓練完成，生成範例
# ─────────────────────────────────────────

print("\n訓練完成，生成範例：\n")
model.eval()
prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
output = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)
print(decode(output[0].tolist()))