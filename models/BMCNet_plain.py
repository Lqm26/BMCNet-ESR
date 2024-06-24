from .submodules import *

class Backbone(nn.Module):
    def __init__(self, n_c, n_b, scale, repeat):
        super(Backbone,self).__init__()
        pad = (1,1)
        self.conv_f1 = nn.Conv2d(scale**2 + n_c + 2*repeat, n_c, 3, 1, padding=pad)
        self.conv_f2 = self.conv_f1
        self.conv_fs = nn.Conv2d(scale**2*2 + n_c + 2*2*repeat, n_c, 3, 1, padding=pad)

        self.para_reschunk = nn.ModuleList([BIE(n_c)] * n_b)
        self.scale = scale

        self.conv_h = nn.Conv2d(n_c, n_c, 3, 1, padding=pad)
        self.conv_o = nn.Conv2d(n_c * 2, scale**2*2, 3, 1, padding=pad)
        
        initialize_weights([self.conv_f1, self.conv_f2, self.conv_h, self.conv_o], 0.1)
        
    def forward(self, xs, h, o):
        x1, x2 = xs
        xs = torch.cat([x1, x2], dim=1)
        o1, o2 = o[:,:self.scale**2], o[:,self.scale**2:]
        x1 = F.relu(self.conv_f1(torch.cat([x1, h, o1], dim=1)))
        x2 = F.relu(self.conv_f2(torch.cat([x2, h, o2], dim=1)))
        xs = F.relu(self.conv_fs(torch.cat([xs, h, o], dim=1)))

        for layer in self.para_reschunk:
            x1, x2, xs = layer(x1, x2, xs)
        x = torch.cat([x1, x2], dim=1)
        x_h = F.relu(self.conv_h(xs))
        x_o = self.conv_o(x)
        
        return x_h, x_o


class BMCNet_plain(nn.Module):
    def __init__(self, scale, n_c, n_b, repeat=3):
        super(BMCNet_plain, self).__init__()
        self.neuro = Backbone(n_c, n_b, scale, repeat=repeat)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.repeat = repeat

    def forward(self, x, x_h, x_o, init):
        """
        Parameters
        ----------
        x: Tensor [B, N, T, H, W], N=3, T=2
        x_h: Tensor [B, N*k*k, H, W]
        x_o: Tensor [B, N*k*k, H, W]
        init: bool
        """

        _, _, T, _, _ = x.shape
        f1 = x[:, :, 0, :, :]
        f2 = x[:, :, 1, :, :]

        x_input_1 = torch.cat((f1[:, 0:1, :, :].repeat(1, self.repeat, 1, 1), f2[:, 0:1, :, :].repeat(1, self.repeat, 1, 1)), dim=1)
        x_input_2 = torch.cat((f1[:, 1:2, :, :].repeat(1, self.repeat, 1, 1), f2[:, 1:2, :, :].repeat(1, self.repeat, 1, 1)), dim=1)

        if init:
            x_h, x_o = self.neuro([x_input_1, x_input_2], x_h, x_o)
        else:
            x_o = self.down(x_o)
            x_h, x_o = self.neuro([x_input_1, x_input_2], x_h, x_o)
        x_o = F.pixel_shuffle(x_o, self.scale) + F.interpolate(f2[:,:2], scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        return x_h, x_o

