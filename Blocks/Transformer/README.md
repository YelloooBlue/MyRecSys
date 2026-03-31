
# 总结


我拆解一下掌握Transformer手搓的要点：

1. 首先是多头注意力，这个逃不脱，也是基础考点，要会搓。
2. 位置编码，能记就记吧。
3. FFN就是每个位置过MLP。

接下来

1. Encoder Layer 记住：
    - x = LN(x + dropout(attn(x))) # 或者PreNorm： x = x + dropout(attn(LN(x)))
    - x = LN(x + dropout(ffn(x)))
2. Encoder:
    - mask padding
    - 位置编码 + dropout
    - 过多层Layer


3. Decoder Layer记住：
    - x = LN(x + dropout(masked_attn(x)))   # causal mask + padding mask
    - x = LN(x + dropout(cross_attn(x)))    # kv用Encoder的最后状态，只需要padding mask
    - x = LN(x + dropout(ffn(x)))
4. Decoder：
    - mask padding
    - mask causal
    - 位置编码 + dropout
    - 过多层Layer

5. 最后线性层投影到词表大小，Softmax输出概率分布，然后使用交叉熵损失训练。


## 必会模块
- Multi-Head Attention 实现
- FFN 的实现

下面两个，来不及就算了（先把框架写出来再解释大概率也是可以接受的）
- Padding Mask 和 Causal Mask 组织
- Position Encoding 实现

## 全程形状

设定：
* **B**: Batch Size（批大小）
* **T**: Sequence Length（序列长度，$T_{s}$ 为源序列，$T_{t}$ 为目标序列）
* **D**: Embedding Dimension（特征维度，$d_{model}$）
* **H**: Number of Heads（头数）
* **V**: Vocab Size（词表大小）

### 1. 基础组件
* **Input Embedding**: $(B, T) \rightarrow (B, T, D)$
* **Positional Encoding**: $(T, D)$，直接加到 Embedding 上，形状保持 $(B, T, D)$。
* **MHA (Multi-Head Attention)**:
    * $Q, K, V$ 输入: $(B, T, D)$
    * Split heads: $(B, H, T, D/H)$
    * Attention Score ($QK^T$): $(B, H, T, T)$
    * Output: 合并头后回到 $(B, T, D)$
* **FFN**: $(B, T, D) \xrightarrow{Linear} (B, T, 4D) \xrightarrow{Linear} (B, T, D)$

### 2. Encoder Layer
1.  **Self-Attention**: 输入 $(B, T_s, D) \rightarrow$ 输出 $(B, T_s, D)$
2.  **Add & Norm**: 始终保持 $(B, T_s, D)$
3.  **FFN**: 始终保持 $(B, T_s, D)$
> **Encoder 最终输出 (Memory)**: $(B, T_s, D)$

### 3. Decoder Layer
1.  **Masked Self-Attention**: 输入 $(B, T_t, D) \rightarrow$ 输出 $(B, T_t, D)$
2.  **Cross-Attention**:
    * **Q** (来自 Decoder): $(B, T_t, D)$
    * **K, V** (来自 Encoder): $(B, T_s, D)$
    * 输出形状: $(B, T_t, D)$（注意：形状随 $Q$ 走，即 $T_t$）
3.  **FFN**: 输出 $(B, T_t, D)$

### 4. 输出投影与损失
1.  **Linear**: $(B, T_t, D) \rightarrow (B, T_t, V)$
2.  **Softmax**: 最后一个维度 $V$ 上做，形状不变 $(B, T_t, V)$
3.  **Loss**: 将预测 $(B \times T_t, V)$ 与 标签 $(B \times T_t)$ 对齐计算。

### 极简总结表

| 步骤 | 输入形状 | 输出形状 |
| :--- | :--- | :--- |
| **Encoder 输入** | $(B, T_s)$ | $(B, T_s, D)$ |
| **Encoder Layer** | $(B, T_s, D)$ | $(B, T_s, D)$ |
| **Decoder 输入** | $(B, T_t)$ | $(B, T_t, D)$ |
| **Cross-Attention** | $Q:(B, T_t, D), K:(B, T_s, D)$ | $(B, T_t, D)$ |
| **最终输出层** | $(B, T_t, D)$ | $(B, T_t, V)$ |