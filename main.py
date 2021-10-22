import os
from tqdm import tqdm
import torch
from torch import nn,optim
import numpy as np
from torch.nn.modules.utils import _triple
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import init
import math
import mdconv

# Apply mdconv kernel
class MDConvFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias,kernel_size, stride, padding, cof):
        ctx.stride = stride
        ctx.padding = padding
        ctx.kernel_size = kernel_size
        ctx.cof = cof
        output = mdconv.forward(input, weight, bias,1,
                                         ctx.kernel_size[0], ctx.kernel_size[1],ctx.kernel_size[2],
                                         ctx.stride[0], ctx.stride[1],ctx.stride[2],
                                         ctx.padding[0], ctx.padding[1],ctx.padding[2],
                                         cof[0],cof[1],cof[2],cof[3])
        ctx.save_for_backward(input, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input,  weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = mdconv.backward(input,weight,bias,grad_output,1,
                                     ctx.kernel_size[0], ctx.kernel_size[1], ctx.kernel_size[2],
                                     ctx.stride[0], ctx.stride[1], ctx.stride[2],
                                     ctx.padding[0], ctx.padding[1], ctx.padding[2],
                                     ctx.cof[0],ctx.cof[1],ctx.cof[2],ctx.cof[3])

        return grad_input, grad_weight, grad_bias,None,None,None,None,None

# Temporal convolution with mode 'up', 'down', 'right' or 'left'
class DirectionalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False,padding=(0,0,0), mode='up'):
        super(DirectionalConv, self).__init__()

        if not torch.cuda.is_available():
            raise EnvironmentError("only support for GPU mode")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.use_bias = bias

        if mode == 'up':
            self.cof = [-1,   kernel_size[0]-1,    0,   0]
            self.padding = (kernel_size[0] // 2-1+padding[0], kernel_size[0] // 2+padding[1], kernel_size[2] // 2+padding[2])
        elif mode == 'down':
            self.cof = [ 1,   0,   0,      0]
            self.padding = (kernel_size[0] // 2-1+padding[0], kernel_size[0] // 2+padding[1], kernel_size[2] // 2+padding[2])
        elif mode == 'right':
            self.cof = [  0,   0,      1,   0 ]
            self.padding = (kernel_size[0] // 2-1+padding[0], kernel_size[1] // 2+padding[1], kernel_size[0] // 2+padding[2])
        elif mode == 'left':
            self.cof = [   0,   0,    -1,   kernel_size[0] - 1]
            self.padding = (kernel_size[0] // 2-1+padding[0], kernel_size[1] // 2+padding[1], kernel_size[0] // 2+padding[2])
        else:
            raise ValueError("no such mode")

        # weight of kernel
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size),requires_grad=True).cuda()
        self.bias = nn.Parameter(torch.zeros(out_channels).float(),requires_grad=True).cuda()
        self.reset_parameters()
        if not self.use_bias:
            self.bias.detach()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return MDConvFunction.apply(input, self.weight,self.bias,
                                    self.kernel_size,self.stride,self.padding,self.cof)

# You can use MDConv instead of Conv3d
class MDConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3,s_padding=(0,1,1),t_padding = (1, 0, 0), stride=1, bias=False, ratial = 0.2, t_downsample = False,first_conv=False):
        super(MDConv,self).__init__()
        if not torch.cuda.is_available():
            raise EnvironmentError("only support for GPU mode")
        per_out_channels = int((1 - ratial) / 4 * out_channels)
        t_kernel_size = (3, 1, 1)
        s_kernel_size = (1, kernel_size, kernel_size)

        if t_downsample:
            t_stride = (stride, 1, 1)
        else:
            t_stride = (1, 1, 1)

        s_stride = (1, stride, stride)

        if first_conv:
            self.spatial = nn.Conv3d(in_channels, out_channels, s_kernel_size, s_stride, s_padding, bias=bias)
            self.bn = nn.BatchNorm3d(out_channels)
            in_channels = out_channels
        else:
            self.spatial = nn.Conv3d(in_channels, in_channels, s_kernel_size, s_stride, s_padding, bias=bias)
            self.bn = nn.BatchNorm3d(in_channels)

        self.up=DirectionalConv(in_channels, per_out_channels,  t_kernel_size, t_stride, bias=bias, padding=t_padding, mode='up')
        self.down = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, bias=bias, padding=t_padding, mode='down')
        self.left = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, bias=bias, padding=t_padding, mode='left')
        self.right = DirectionalConv(in_channels, per_out_channels, t_kernel_size, t_stride, bias=bias, padding=t_padding, mode='right')
        self.relu = nn.ReLU()

    def forward(self,x):
        s = self.relu(self.bn(self.spatial(x)))
        x1 = self.up(s)
        x2 = self.down(x)
        x3 = self.left(x)
        x4 = self.right(x)
        x=torch.cat([x1,x2,x3,x4],1)
        return x
