import torch
import torch.nn as nn
import torch.nn.functional as F


def SigmoidFunc(x, alpha=3.):   # Sigmoid function with parameter
    return torch.sigmoid(alpha * x)

def SigmoidGFunc(x, alpha=3.):   # Derivative of Sigmoid functions with parameter
    return alpha * SigmoidFunc(x, alpha) * (1 - SigmoidFunc(x, alpha))

def BinaryForFunc(x):   # Weight quantization forward function
    return torch.sign(x)

def BinaryBackFunc(x, alpha=3.):   # Quantization backward gradient function
    return 2 * SigmoidGFunc(x, alpha)

class SignFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return BinaryForFunc(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * BinaryBackFunc(input, alpha=3.)
    
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()
        
    def forward(self, x):
        ba = SignFunc.apply(x)
        return ba

class Q_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class Q_PReLU(nn.Module):
    def __init__(self, out_chn):
        super(Q_PReLU, self).__init__()
        self.L_alpha = nn.Parameter(-3 * torch.ones(out_chn), requires_grad=True)

    def forward(self, x):
        IL_alpha = Q_Func.apply(self.L_alpha)
        QIL_alpha = torch.pow(input = torch.tensor(2).to(x.device), exponent = IL_alpha)
        return F.prelu(x, QIL_alpha)

def get_weight(module):
    std, mean = torch.std_mean(module.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
    weight = (module.weight - mean) / (std + module.eps)
    return weight

def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class ScaledStdConv2d(nn.Conv2d):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1,
            bias=False, gamma=1.0, eps=1e-5, use_layernorm=False):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm  # experimental, slightly faster/less GPU memory to hijack LN kernel

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class HardBinaryScaledStdConv2d(nn.Module):

    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, gamma=1.0, eps=1e-5, use_layernorm=False):
        super(HardBinaryScaledStdConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand(self.shape) * 0.001, requires_grad=True)

        self.gain = nn.Parameter(torch.ones(out_chn, 1, 1, 1))
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps = eps ** 2 if use_layernorm else eps
        self.use_layernorm = use_layernorm

    def get_weight(self):
        if self.use_layernorm:
            weight = self.scale * F.layer_norm(self.weight, self.weight.shape[1:], eps=self.eps)
        else:
            std, mean = torch.std_mean(self.weight, dim=[1, 2, 3], keepdim=True, unbiased=False)
            weight = self.scale * (self.weight - mean) / (std + self.eps)

        scaling_factor = torch.mean(torch.mean(torch.mean(abs(weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(weight)
        cliped_weights = torch.clamp(weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        return self.gain * binary_weights

    def forward(self, x):

        return F.conv2d(x, self.get_weight(), stride=self.stride, padding=self.padding)