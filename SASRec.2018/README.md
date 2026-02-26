# SASRec模型复现（PyTorch）
SASRec（Self-Attentive Sequential Recommendation）是一种基于自注意力机制的序列推荐模型，旨在捕捉用户历史交互序列中的长期依赖关系。本文档介绍了SASRec模型的复现过程，包括数据集准备、数据预处理、构建交互序列数据集以及模型训练等步骤。

TAGS: 召回、序列推荐、Transformer、自注意力机制

## 数据集准备
TBD

## 数据预处理
TBD

## 构建数据集（交互序列）
TBD

## 训练
使用 `SASRec.2018/train.py` 脚本训练SASRec模型。

```bash
  cd SASRec.2018/
  python train.py
```

## 评估
使用 `test.py` 脚本评估模型性能，进行FAISS检索评估。

### Reference
- 论文链接：[Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- 原作者代码：[SASRec（TensorFlow）](https://github.com/kang205/SASRec)
- 社区实现：[SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch)