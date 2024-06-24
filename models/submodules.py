import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torch.nn import init
import numpy as np
import functools


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class BIE(nn.Module):
    def __init__(self, nf=64):
        super(BIE, self).__init__()
        # self-process
        self.conv1 = ResidualBlock_noBN(nf)
        self.conv2 = self.conv1
        self.convf1 = nn.Conv2d(nf * 2, nf, 1, 1, padding=0)
        self.convf2 = self.convf1

        self.scale = nf ** -0.5
        self.norm_s = LayerNorm2d(nf)
        self.clustering = nn.Conv2d(nf, nf, 1, 1, padding=0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, stride=1, padding=0)

        self.v1 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)
        self.v2 = nn.Conv2d(nf, nf, 1, stride=1, padding=0)

        # initialization
        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering, self.v1, self.v2], 0.1)

    def forward(self, x_1, x_2, x_s):
        b, c, h, w = x_1.shape

        x_1_ = self.conv1(x_1)
        x_2_ = self.conv2(x_2)
        shared_class_center1 = self.clustering(self.norm_s(self.convf1(torch.cat([x_s, x_2], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]
        shared_class_center2 = self.clustering(self.norm_s(self.convf2(torch.cat([x_s, x_1], dim=1)))).view(b, c, -1) # [b, c, h, w] -> [b, c, h*w]

        v_1 = self.v1(x_1).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]
        v_2 = self.v2(x_2).view(b,c,-1).permute(0, 2, 1)  # [b, c, h, w] -> [b, c, hw] -> [b, hw, c]

        att1 = torch.bmm(shared_class_center1, v_1) * self.scale # [b, c, hw] x [b, hw, c] -> [b, c, c]
        att2 = torch.bmm(shared_class_center2, v_2) * self.scale  # [b, c, hw] x [b, hw, c] -> [b, c, c]

        out_1 = torch.bmm(torch.softmax(att1, dim=-1), v_1.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]
        out_2 = torch.bmm(torch.softmax(att2, dim=-1), v_2.permute(0, 2, 1)).view(b, c, h, w)  # [b, c, c] x [b, c, hw] -> [b, c, hw]

        x_s_ = self.unclustering(torch.cat([shared_class_center1.view(b, c, h, w), shared_class_center2.view(b, c, h, w)], dim=1)) + x_s

        return out_1 + x_2_, out_2 + x_1_, x_s_


def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

