import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 测试数据：覆盖正负值、极值
x = torch.tensor([-10, -1, 0, 1, 10], dtype=torch.float32)

def test_activations(x):
    # -------------------------- 手动实现（数学公式版） --------------------------
    # ReLU
    relu_manual = torch.max(x, torch.zeros_like(x))

    # Leaky ReLU
    negative_slope=0.01
    leaky_relu_manual = torch.where(x>0, x, negative_slope*x)

    # PReLU
    weight = 0.2    # PReLU 的可学习参数，这里我们手动设置为0.2
    prelu_manual = torch.where(x>0, x, weight*x)

    # ELU
    alpha = 1.0
    elu_manual = torch.where(x>0, x, alpha*(torch.exp(x)-1))

    # Sigmoid
    sigmoid_manual = 1 / (1 + torch.exp(-x))

    # Tanh
    tanh_manual = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

    # GELU（近似公式）
    gelu_manual = 0.5 * x * (
        1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715*x**3))
    )

    # Swish 也被称为 SiLU，在 PyTorch 中是 SiLU 且 beta固定为1 变为了 Sigmoid 函数
    swish_manual = x * torch.sigmoid(x) 

    # Softmax（数值稳定版）
    max_x = torch.max(x)
    exp_x = torch.exp(x - max_x)
    softmax_manual = exp_x / torch.sum(exp_x)

    # Mish
    mish_manual = x * torch.tanh(F.softplus(x))

    # -------------------------- PyTorch API 简洁实现 --------------------------
    relu_api = nn.ReLU()(x)
    leaky_relu_api = nn.LeakyReLU(negative_slope=0.01)(x)
    prelu_api = nn.PReLU(num_parameters=1, init=0.2)(x)
    elu_api = nn.ELU(alpha=1.0)(x)
    sigmoid_api = nn.Sigmoid()(x)
    tanh_api = nn.Tanh()(x)
    gelu_api = nn.GELU()(x)
    swish_api = nn.SiLU()(x)    # Swish 在 PyTorch 中是 SiLU
    softmax_api = nn.Softmax(dim=-1)(x)
    mish_api = nn.Mish()(x)

    # -------------------------- 输出对比 --------------------------
    print("="*50)
    print("输入：", x.numpy())
    print("="*50)

    print("ReLU 手动：", relu_manual.numpy())
    print("ReLU API ：", relu_api.detach().numpy())
    print()

    print("Leaky ReLU 手动：", leaky_relu_manual.numpy())
    print("Leaky ReLU API ：", leaky_relu_api.detach().numpy())
    print()

    print("PReLU 手动：", prelu_manual.numpy())
    print("PReLU API ：", prelu_api.detach().numpy())
    print()

    print("ELU 手动：", elu_manual.numpy())
    print("ELU API ：", elu_api.detach().numpy())
    print()

    print("Sigmoid 手动：", sigmoid_manual.numpy())
    print("Sigmoid API ：", sigmoid_api.detach().numpy())
    print()

    print("Tanh 手动：", tanh_manual.numpy())
    print("Tanh API ：", tanh_api.detach().numpy())
    print()

    print("GELU 手动：", gelu_manual.numpy())
    print("GELU API ：", gelu_api.detach().numpy())
    print()

    print("Swish 手动：", swish_manual.numpy())
    print("Swish API ：", swish_api.detach().numpy())
    print()

    print("Softmax 手动：", softmax_manual.numpy())
    print("Softmax API ：", softmax_api.detach().numpy())
    print()

    print("Mish 手动：", mish_manual.numpy())
    print("Mish API ：", mish_api.detach().numpy())

    # -------------------------- 画激活函数曲线 --------------------------
    x_plot = torch.linspace(-10, 10, 400)

    relu = torch.maximum(x_plot, torch.zeros_like(x_plot))
    leaky_relu = torch.where(x_plot>0, x_plot, 0.01*x_plot)
    prelu = torch.where(x_plot>0, x_plot, 0.2*x_plot)
    elu = torch.where(x_plot>0, x_plot, torch.exp(x_plot)-1)
    sigmoid = 1/(1+torch.exp(-x_plot))
    tanh = (torch.exp(x_plot)-torch.exp(-x_plot))/(torch.exp(x_plot)+torch.exp(-x_plot))
    gelu = 0.5*x_plot*(1+torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x_plot+0.044715*x_plot**3)))
    swish = x_plot*torch.sigmoid(x_plot)
    mish = x_plot*torch.tanh(torch.nn.functional.softplus(x_plot))

    plt.figure(figsize=(12,8))

    plt.plot(x_plot.numpy(), relu.numpy(), label="ReLU")
    plt.plot(x_plot.numpy(), leaky_relu.numpy(), label="LeakyReLU")
    plt.plot(x_plot.numpy(), prelu.numpy(), label="PReLU")
    plt.plot(x_plot.numpy(), elu.numpy(), label="ELU")
    plt.plot(x_plot.numpy(), sigmoid.numpy(), label="Sigmoid")
    plt.plot(x_plot.numpy(), tanh.numpy(), label="Tanh")
    plt.plot(x_plot.numpy(), gelu.numpy(), label="GELU")
    plt.plot(x_plot.numpy(), swish.numpy(), label="Swish")
    plt.plot(x_plot.numpy(), mish.numpy(), label="Mish")

    plt.axhline(0)
    plt.axvline(0)

    plt.legend()
    plt.title("Activation Functions")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)

    plt.show()

# 执行测试
test_activations(x)