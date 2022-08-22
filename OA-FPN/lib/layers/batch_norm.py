import torch
from torch import nn

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    """

    def __init__(self, num_features, eps=1e-5):  #num_features是输入的通道数，eps就是一个很小的数ε，防止分母为零
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))     #feture map满足均值为0，方差为1的分布规律
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()   #rsqrt函数是取张量平方根的倒数
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)  #reshape(-1)代表暂不确定这一维度是多少，reshape(-1,1)就是不管多少行，反正就1列
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias   # （x-均值）/（根号下方差+ε）

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

