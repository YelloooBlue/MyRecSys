# 常见激活函数实现
import torch
import torch.nn as nn

# =============================================== ReLU 家族 ===============================================

# ReLU（Rectified Linear Unit）函数是深度学习中最常用的激活函数之一，定义为 f(x) = max(0, x)。它的优点是计算简单且能够有效地缓解梯度消失问题。
class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return torch.max(x, torch.zeros_like(x))
    
# Leaky ReLU（Leaky Rectified Linear Unit）是ReLU的一个变体，定义为 f(x) = max(ax, x)，其中a是一个小的常数（通常为0.01）。Leaky ReLU在处理负输入时引入了一个小的斜率，避免了ReLU在负输入时完全没有梯度的问题。
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.where(x > 0, x, self.negative_slope * x)

# PReLU（Parametric ReLU）是ReLU的一个变体，定义为 f(x) = max(ax, x)，其中a是一个可学习的参数。PReLU允许模型在训练过程中自动调整负输入的斜率，从而提高模型的表达能力。
class PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))

    def forward(self, x):
        return torch.where(x > 0, x, self.weight * x)

# ELU（Exponential Linear Unit）是ReLU的一个变体，定义为 f(x) = x if x > 0 else alpha * (exp(x) - 1)，其中alpha是一个常数（通常为1）。ELU在处理负输入时引入了一个指数函数，可以使输出更平滑，并且在负输入时具有非零梯度。
class ELU(nn.Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

# =============================================== Sigmoid 和 Tanh ===============================================
    
# Sigmoid函数将输入映射到(0, 1)之间，定义为 f(x) = 1 / (1 + exp(-x))。它常用于二分类问题的输出层，以表示某个样本属于正类的概率。
class Sigmoid(nn.Module):
    """
    Sigmoid 某种程度上可以被视为「对数几率」函数（log（p / (1 - p))）的逆函数，他们俩将数值在(-∞, +∞)和(0, 1)之间进行映射。
    其中(-∞, +∞)也就是我们常说的logits，(0, 1)则是概率值。Sigmoid函数在二分类问题中非常常见，因为它可以将输出映射到一个概率值，表示某个样本属于正类的概率。
    """
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))
    
class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    
# =============================================== Softmax ===============================================

# Softmax函数将输入映射到(0, 1)之间，并且所有输出的和为1，定义为 f(x_i) = exp(x_i) / sum(exp(x_j))。它常用于多分类问题的输出层，以表示某个样本属于每个类别的概率分布。
class Softmax(nn.Module):
    """
    为什么Softmax要在exp空间进行计算？这是为了确保输出的概率值都是正数，并且总和为1。
    直接在原始空间进行计算可能会导致数值不稳定，尤其是当输入值较大或较小时，可能会出现溢出或下溢的情况。
    """
    def __init__(self, dim=-1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        exp_x = torch.exp(x - torch.max(x, dim=self.dim, keepdim=True)[0])  # 稳定性处理
        return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)