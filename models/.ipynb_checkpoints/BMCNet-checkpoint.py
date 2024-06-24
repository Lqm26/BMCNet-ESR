from .submodules import *

class ParallelBlk(nn.Module):
    def __init__(self, nf=64):
        super(ParallelBlk, self).__init__()
        self.conv1 = ResidualBlock_noBN(nf)
        self.conv2 = self.conv1
        self.conv1_st = ResidualBlock_noBN(nf)
        self.conv2_st = self.conv1_st

        ### local BIE
        self.lBIE = BIE(nf)
        ### global BIE
        self.gBIE = BIE(nf)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv1_st, self.conv2_st], 0.1)

    def forward(self, x_1, x_2, x_s, x_1_st, x_2_st, x_1_s_st, x_2_s_st):

        x_1 = self.conv1(x_1)
        x_2 = self.conv2(x_2)

        x_1_st = self.conv1_st(x_1_st)
        x_2_st = self.conv2_st(x_2_st)

        x_1, x_1_st, x_1_s_st = self.lBIE(x_1, x_1_st, x_1_s_st)
        x_2, x_2_st, x_2_s_st = self.lBIE(x_2, x_2_st, x_2_s_st)

        x_1, x_2, out_s = self.gBIE(x_1, x_2, x_s)

        return x_1, x_2, out_s, x_1_st, x_2_st, x_1_s_st, x_2_s_st


class Backbone(nn.Module):
    def __init__(self, n_c, n_b, scale, repeat):
        super(Backbone, self).__init__()
        pad = (1, 1)
        
        self.conv_fpst = nn.Conv2d(scale**2 + n_c + 2*repeat, n_c, 3, 1, padding=pad)
        self.conv_fnst = self.conv_fpst
        self.conv_fps = nn.Conv2d(repeat + n_c, n_c, 3, 1, padding=pad)
        self.conv_fns = self.conv_fps
        self.conv_fs = nn.Conv2d(scale**2*2 + n_c * 3, n_c, 3, 1, padding=pad)

        self.para_reschunk = nn.ModuleList([ParallelBlk(n_c)] * n_b)
        self.scale = scale

        self.conv_hs = nn.Conv2d(n_c, n_c, 3, 1, padding=pad)
        self.conv_hp = nn.Conv2d(n_c, n_c, 3, 1, padding=pad)
        self.conv_hn = nn.Conv2d(n_c, n_c, 3, 1, padding=pad)

        self.conv_o = nn.Conv2d(n_c * 2, scale**2*2, 3, 1, padding=pad)

        initialize_weights([self.conv_fpst, self.conv_fnst, self.conv_fps, self.conv_fns, self.conv_fs, self.conv_hs, self.conv_hp, self.conv_hn, self.conv_o], 0.1)

    def forward(self, xs, hp, hn, hs, o):
        x1p, x1n, x2p, x2n = xs

        xp = torch.cat([x1p, x2p], dim=1)
        xn = torch.cat([x1n, x2n], dim=1)

        op, on = o[:,:self.scale**2], o[:,self.scale**2:]
        xp_st = F.relu(self.conv_fpst(torch.cat([xp, hp, op], dim=1)))
        xn_st = F.relu(self.conv_fnst(torch.cat([xn, hn, on], dim=1)))
        xp_s = F.relu(self.conv_fps(torch.cat([x2p, hp], dim=1)))
        xn_s = F.relu(self.conv_fns(torch.cat([x2n, hn], dim=1)))

        xs_ = torch.cat([xp_st, xn_st], dim=1)
        xs = F.relu(self.conv_fs(torch.cat([xs_, hs, o], dim=1)))

        xs_p_st = F.relu(self.conv_fs(torch.cat([xs_, hp, o], dim=1)))
        xs_n_st = F.relu(self.conv_fs(torch.cat([xs_, hn, o], dim=1)))

        for layer in self.para_reschunk:
            xp_s, xn_s, xs, xp_st, xn_st, xs_p_st, xs_n_st = layer(xp_s, xn_s, xs, xp_st, xn_st, xs_p_st, xs_n_st)
            
        x = torch.cat([xp_s, xn_s], dim=1)
        x_h = F.relu(self.conv_hs(xs))
        x_h_p = F.relu(self.conv_hp(xs_p_st))
        x_h_n = F.relu(self.conv_hn(xs_n_st))
        x_o = self.conv_o(x)
        
        return x_h, x_h_p, x_h_n, x_o


class BMCNet(nn.Module):
    def __init__(self, scale, n_c, n_b, repeat=3):
        super(BMCNet, self).__init__()
        self.neuro = Backbone(n_c, n_b, scale, repeat=repeat)
        self.scale = scale
        self.down = PixelUnShuffle(scale)
        self.repeat = repeat

    def forward(self, x, x_h, x_h_p, x_h_n, x_o, init):
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

        x_input_1_p = f1[:, 0:1, :, :].repeat(1, self.repeat, 1, 1)
        x_input_1_n = f1[:, 1:2, :, :].repeat(1, self.repeat, 1, 1)
        x_input_2_p = f2[:, 0:1, :, :].repeat(1, self.repeat, 1, 1)
        x_input_2_n = f2[:, 1:2, :, :].repeat(1, self.repeat, 1, 1)

        if init:
            x_h, x_h_p, x_h_n, x_o = self.neuro([x_input_1_p, x_input_1_n, x_input_2_p, x_input_2_n], x_h, x_h_p, x_h_n, x_o)
        else:
            x_o = self.down(x_o)
            x_h, x_h_p, x_h_n, x_o = self.neuro([x_input_1_p, x_input_1_n, x_input_2_p, x_input_2_n], x_h, x_h_p, x_h_n, x_o)
        x_o = F.pixel_shuffle(x_o, self.scale) + F.interpolate(f2[:,:2], scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        return x_h, x_h_p, x_h_n, x_o

