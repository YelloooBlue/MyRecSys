

## 模型列表
| 模型名称 | 年份 | 论文链接 | 说明 | 类型 |
| -------- | ------- | ---- | ---- | ---- |
| [DIN](/DIN.2017/README.md) | 2017 | [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) | 阿里巴巴提出，适用于CTR预估任务，候选广告感知的兴趣建模（target attention） | 排序 |
| [DIEN](/DIEN.2018/README.md) | 2018 | [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672) | 阿里巴巴提出，适用于CTR预估任务，改进了DIN中兴趣演化建模的能力，显式建模兴趣的动态变化 | 排序 |
| [DSIN](/DSIN.2019/README.md) | 2019 | [Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1905.06482) | 阿里巴巴提出，适用于CTR预估任务，将用户行为划分为 Session，建模分层兴趣建模（session-level + session evolution） | 排序 |
| [SIM](/SIM.2020/README.md) | 2020 | [Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction](https://arxiv.org/abs/2006.05639) | 阿里巴巴提出，适用于CTR预估任务，解决超长用户行为序列问题，使用两阶段检索机制捕捉兴趣演变 | 排序 |
| [SASRec](/SASRec.2018/README.md) | 2018 | [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) | 基于Transformer的序列推荐模型，适用于捕捉用户历史交互序列中的长期依赖关系 | 序列推荐 |
|BERT4Rec||||序列推荐|
|GRU4Rec||||序列推荐|
|OneRec|||快手|端到端生成式推荐|
|OneRecv2|||快手|端到端生成式推荐|
|HSTU|||Meta提出生成式推荐模型||
|MTGR|||美团提出生成式+传统模型||
|MIND|||多兴趣召回|召回|
|DSSM|||双塔结构，分别学习用户和物品向量，通过向量相似度进行匹配，适合大规模向量检索|召回|
|Item2Vec|||基于 Word2Vec 思想，将物品共现序列转化为 embedding，用于学习物品相似度|表征学习、i2i召回|
|YouTube DNN|||两阶段推荐框架中的召回模型，使用多层神经网络预测用户对候选视频的概率|召回|
|Wide&Deep|||结合线性模型（记忆能力）和深度网络（泛化能力）的结构|排序|
|DeepFM|||将 FM 与 DNN 结合，实现自动特征交叉学习|排序|
|MMoE|||多门控专家网络，用于多任务学习，通过门控机制实现任务间参数共享|多任务排序|
|DCN|||通过 Cross Network 显式建模高阶特征交叉|排序|
|ESMM|||解决多阶段转化率预估（CTR + CVR）样本选择偏差问题|多任务排序|
|PLE|||渐进式分层专家网络，改进多任务负迁移问题|多任务排序|
|FM|||因子分解机，建模二阶特征交叉，适用于稀疏特征场景|排序 / 特征交叉|