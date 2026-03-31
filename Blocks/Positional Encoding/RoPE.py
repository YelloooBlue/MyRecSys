import torch

def rope(x):
    """
    x: [B, H, L, D]
        B：batch_size（无关）
        H：num_heads（无关）
        L：seq_len
        D：embedding_dim

    大致流程：
    以序列中每个位置为例：
        - 原向量：[x0, x1, x2, x3] D = 4
        - 分组：[(x0, x1), (x2, x3)]
        - 旋转：[(x0', x1'), (x2', x3')]
        - 拼接：[x0', x1', x2', x3']

    """
    B, H, L, D = x.shape
    assert D % 2 == 0       # 因为要两两一组进行旋转，所以维度必须是偶数

    # ===== 生成 RoPE 旋转矩阵 =====

    # 生成序列长度的 position
    pos = torch.arange(L, dtype=torch.float32, device=x.device)  # [L] = [0, 1, 2, ..., L-1]

    # 为向量的每一对维度计算一个频率
    dim = torch.arange(start=0, end=D, step=2, dtype=torch.float32, device=x.device)  # [D/2] = [0, 2, 4, ..., D-2]
    inv_freq = 1.0 / (10000 ** (dim / D))  # [D/2] = [1, 1/10000^(2/D), 1/10000^(4/D), ..., 1/10000^((D-2)/D)]

    # 角度矩阵（每个位置分配一个角度）
    sinusoid = torch.einsum("l,d->l d", pos, inv_freq)  # [L] x [D/2] -> [L, D/2]

    """
        直观理解：
        - 序列位置 (pos) = 走过的“时间/步数”：Token 位置越靠后，经历的时间越长，累积的总旋转角度越大。
        - 向量维度 (dim) = 指针的“基础转速”：
            - 维度越低（靠前），频率越高，转得越快（如秒针）
            - 维度越高（靠后），频率越低，转得越慢（如时针）
        
        不怕「序列靠后 + 维度靠后」的值与「序列靠前 + 维度靠前」的值重复吗？
            不怕，因为在 Attention 点积计算时，不同维度之间是严格对齐且独立内积的（第 d 维只与第 d 维交互）。
            即使绝对角度偶然相同，它们也处于互不干涉的“平行宇宙”中。
        
        通过外积（einsum），我们将“步数”与“转速”相乘，得到一个 [L, D/2] 的角度（相位）矩阵：
            sinusoid[l, d] = pos[l] * inv_freq[d]  (即：总角度 = 步数 × 转速)
    """

    # ===== 应用 RoPE 旋转 =====

    sin = sinusoid.sin()[None, None, :, :]  # [1, 1, L, D/2]
    cos = sinusoid.cos()[None, None, :, :]  # [1, 1, L, D/2]

    # 拆偶数奇数维
    x_even = x[..., 0::2]   # [B, H, L, D]-> [B, H, L, D/2]
    x_odd = x[..., 1::2]    # [B, H, L, D]-> [B, H, L, D/2]
  
    # 旋转并交错还原形状
    x_rot = torch.empty_like(x)
    x_rot[..., 0::2] = x_even * cos - x_odd * sin   # 填入偶数维
    x_rot[..., 1::2] = x_even * sin + x_odd * cos   # 填入奇数维

    return x_rot