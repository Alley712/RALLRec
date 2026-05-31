"""
训练 LightGCN 获取协同 embedding (e_colla)。
优化版：稀疏矩阵做图卷积，向量化 BPR loss，200 epoch 约 30 分钟。

输入：RALLRec/data/ml-1m/proc_data/train.txt
输出：data/ml-1m/saved_embed/lightgcn_{user,item}_emb.npy + lightgcn_id2item.json
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, eye
import json, os, random, time

# ===== 1. 加载数据 =====
users, items = [], []
with open('/root/autodl-tmp/RALLRec/data/ml-1m/proc_data/train.txt') as f:
    for line in f:
        u, i, _ = line.strip().split()
        users.append(int(u) - 1)
        items.append(int(i) - 1)

num_users = max(users) + 1
num_items = max(items) + 1
num_interactions = len(users)
print(f"Users: {num_users}, Items: {num_items}, Interactions: {num_interactions}")

# ===== 2. 构建归一化邻接矩阵（标准 LightGCN 实现）=====
# R: (num_users, num_items) 交互矩阵
R = coo_matrix((np.ones(num_interactions), (users, items)),
               shape=(num_users, num_items), dtype=np.float32)

# 度矩阵
user_deg = np.array(R.sum(axis=1)).flatten()  # (num_users,)
item_deg = np.array(R.sum(axis=0)).flatten()  # (num_items,)
user_deg[user_deg == 0] = 1
item_deg[item_deg == 0] = 1

# D_u^{-1/2} @ R @ D_i^{-1/2}
user_deg_sqrt_inv = 1.0 / np.sqrt(user_deg)
item_deg_sqrt_inv = 1.0 / np.sqrt(item_deg)
R_norm = R.tocoo()
R_norm.data = user_deg_sqrt_inv[R_norm.row] * item_deg_sqrt_inv[R_norm.col]

# 转为 PyTorch 稀疏张量（GPU）
indices = torch.tensor(np.vstack([R_norm.row, R_norm.col]), dtype=torch.long).cuda()
values = torch.tensor(R_norm.data, dtype=torch.float32).cuda()
R_sparse = torch.sparse_coo_tensor(indices, values, (num_users, num_items)).cuda()

# 交互字典用于负采样
user_items = {}
for u, i in zip(users, items):
    user_items.setdefault(u, set()).add(i)

# ===== 3. LightGCN 传播（稀疏矩阵乘法） =====
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=128, n_layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def propagate(self, u, i):
        """一次全图传播：并行更新，U 和 I 用同一层输入"""
        i_new = torch.sparse.mm(R_sparse.t(), u)  # (items, dim)
        u_new = torch.sparse.mm(R_sparse, i)      # (users, dim)
        return u_new, i_new

    def forward(self, n_layers=3):
        u_emb = self.user_emb.weight   # layer 0
        i_emb = self.item_emb.weight
        all_u, all_i = [u_emb], [i_emb]
        for _ in range(n_layers):
            u_emb, i_emb = self.propagate(u_emb, i_emb)  # 用上一层输出
            all_u.append(u_emb)
            all_i.append(i_emb)
        return sum(all_u) / (n_layers + 1), sum(all_i) / (n_layers + 1)


# ===== 4. 训练（多 mini-batch per epoch） =====
model = LightGCN(num_users, num_items, dim=128).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 1024
batches_per_epoch = 32  # 每 epoch 做 32 次梯度更新，总共 200×32=6400 步

for epoch in range(200):
    t0 = time.time()
    model.train()
    epoch_loss = 0

    for _ in range(batches_per_epoch):
        opt.zero_grad()

        # 采样 mini-batch
        idx = np.random.choice(num_interactions, batch_size, replace=False)
        u_batch = torch.tensor([users[i] for i in idx], dtype=torch.long).cuda()
        pos_batch = torch.tensor([items[i] for i in idx], dtype=torch.long).cuda()

        # 负采样
        neg_batch = torch.randint(0, num_items, (batch_size,), dtype=torch.long).cuda()
        for j in range(batch_size):
            u = u_batch[j].item()
            while neg_batch[j].item() in user_items.get(u, set()):
                neg_batch[j] = random.randint(0, num_items - 1)

        # 一次前向传播
        u_emb, i_emb = model()

        # 向量化 BPR loss
        u_vec = u_emb[u_batch]
        pos_vec = i_emb[pos_batch]
        neg_vec = i_emb[neg_batch]

        pos_score = (u_vec * pos_vec).sum(dim=1)
        neg_score = (u_vec * neg_vec).sum(dim=1)
        loss = -F.logsigmoid(pos_score - neg_score).mean()

        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    elapsed = time.time() - t0
    print(f"Epoch {epoch:3d} | loss={epoch_loss / batches_per_epoch:.4f} | {elapsed:.1f}s/epoch")

# ===== 5. 保存 =====
save_dir = '/root/autodl-tmp/RALLRec/data/ml-1m/saved_embed'
os.makedirs(save_dir, exist_ok=True)

u_emb, i_emb = model()
np.save(os.path.join(save_dir, 'lightgcn_user_emb.npy'), u_emb.cpu().detach().numpy())
np.save(os.path.join(save_dir, 'lightgcn_item_emb.npy'), i_emb.cpu().detach().numpy())

id2item = {str(i): str(i + 1) for i in range(num_items)}
json.dump(id2item, open(os.path.join(save_dir, 'lightgcn_id2item.json'), 'w'))
print(f"\nDone. Saved to {save_dir}/")
print(f"  user_emb:  {u_emb.shape}")
print(f"  item_emb:  {i_emb.shape}")
