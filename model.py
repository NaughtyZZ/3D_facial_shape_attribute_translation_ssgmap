import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6, all_grid_item=None):
        super(Generator, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if all_grid_item is not None:
            self.i1 = all_grid_item[0].to(self.device)
            self.i2 = all_grid_item[1].to(self.device)
            self.i3 = all_grid_item[2].to(self.device)
            self.w1 = all_grid_item[3].to(self.device)
            self.w2 = all_grid_item[4].to(self.device)
            self.w3 = all_grid_item[5].to(self.device)
            self.grid_xy = all_grid_item[6].to(self.device)



        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        # for i in range(2):
        #     layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=1, padding=1, bias=False))
        #     layers.append(nn.PixelShuffle(2))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     curr_dim = curr_dim // 2
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            # layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            # layers.append(nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, c, r):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        # x = torch.bmm(x, r)


        s1 = torch.index_select(x, dim=1, index=self.i1)
        s2 = torch.index_select(x, dim=1, index=self.i2)
        s3 = torch.index_select(x, dim=1, index=self.i3)

        w1_batch = self.w1.view(1, -1, 1).repeat(x.size(0), 1, x.size(2))
        w2_batch = self.w2.view(1, -1, 1).repeat(x.size(0), 1, x.size(2))
        w3_batch = self.w3.view(1, -1, 1).repeat(x.size(0), 1, x.size(2))

        x = w1_batch * s1 + w2_batch * s2 + w3_batch * s3
        x = x.permute(0, 2, 1)
        x = x.view(x.size(0), x.size(1), 128, -1)
        # set mask values
        x[:, :, 56: 72, 92: 97] = 0
        x[:, :, 33: 40, 33: 38] = 0
        x[:, :, 88: 95, 33: 38] = 0
        # x[:, :, 92: 97, 56: 72] = 0
        # x[:, :, 33: 38, 33: 40] = 0
        # x[:, :, 33: 38, 88: 95] = 0

        # x_sym = torch.flip(x, dims=[2])
        # xx_sym_error = x[:, 0, :, :] + x_sym[:, 0, :, :]
        # yy_sym_error = x[:, 1, :, :] - x_sym[:, 1, :, :]
        # zz_sym_error = x[:, 2, :, :] - x_sym[:, 2, :, :]
        # print(xx_sym_error[0, :, :])
        # print(yy_sym_error[0, :, :])
        # print(zz_sym_error[0, :, :])
        # print(torch.mean(xx_sym_error))
        # print(torch.mean(yy_sym_error))
        # print(torch.mean(zz_sym_error))
        # input()

        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        x = self.main(x)
        x_rot_norm = torch.bmm(x.permute(0, 2, 3, 1).view(x.size(0), -1, x.size(1)), r.permute(0, 2, 1))
        x_rot_norm = x_rot_norm.permute(0, 2, 1).view(x.size(0), x.size(1), x.size(2), x.size(3))
        x_rot_norm[:, :, 56: 72, 92: 97] = 0
        x_rot_norm[:, :, 33: 40, 33: 38] = 0
        x_rot_norm[:, :, 88: 95, 33: 38] = 0
        x_sym = torch.flip(x_rot_norm, dims=[2])
        x_sym[:, 0, :, :] = x_rot_norm[:, 0, :, :] + x_sym[:, 0, :, :]
        x_sym[:, 1, :, :] = x_rot_norm[:, 1, :, :] - x_sym[:, 1, :, :]
        x_sym[:, 2, :, :] = x_rot_norm[:, 2, :, :] - x_sym[:, 2, :, :]
        sym_error_sign = c[:, 22, :, :].unsqueeze(1).repeat(1, x_sym.size(1), 1, 1)
        sym_error = x_sym * sym_error_sign

        grid_xy_batch = self.grid_xy.view(1, 1, self.grid_xy.size(0), self.grid_xy.size(1)).repeat(x.size(0), 1, 1, 1)
        out = F.grid_sample(x, grid_xy_batch, mode='bilinear', padding_mode='border', align_corners=False)
        out = out.squeeze(2).permute(0, 2, 1)

        # out=torch.bmm(out, r.permute(0,2,1))
        return out, sym_error


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, all_grid_item=None):
        super(Discriminator, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if all_grid_item is not None:
            self.i1 = all_grid_item[0].to(self.device)
            self.i2 = all_grid_item[1].to(self.device)
            self.i3 = all_grid_item[2].to(self.device)
            self.w1 = all_grid_item[3].to(self.device)
            self.w2 = all_grid_item[4].to(self.device)
            self.w3 = all_grid_item[5].to(self.device)
            self.grid_xy = all_grid_item[6].to(self.device)

        layers1 = []
        layers1.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers1.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num - 4):
            layers1.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers1.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        curr_dim_p1 = curr_dim
        self.main1 = nn.Sequential(*layers1)

        layers2 = []
        for i in range(repeat_num - 4, repeat_num - 2):
            layers2.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers2.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        curr_dim_p2 = curr_dim
        self.main2 = nn.Sequential(*layers2)

        layers3 = []
        for i in range(repeat_num - 2, repeat_num):
            layers3.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers3.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        curr_dim_p3 = curr_dim
        self.main3 = nn.Sequential(*layers3)

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.conv_binary1 = nn.Conv2d(curr_dim_p1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_binary2 = nn.Conv2d(curr_dim_p2, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_binary3 = nn.Conv2d(curr_dim_p3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_cls = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        s1 = torch.index_select(x, dim=1, index=self.i1)
        s2 = torch.index_select(x, dim=1, index=self.i2)
        s3 = torch.index_select(x, dim=1, index=self.i3)

        w1_batch = self.w1.view(1, -1, 1).repeat(x.size(0), 1, x.size(2))
        w2_batch = self.w2.view(1, -1, 1).repeat(x.size(0), 1, x.size(2))
        w3_batch = self.w3.view(1, -1, 1).repeat(x.size(0), 1, x.size(2))

        x = w1_batch * s1 + w2_batch * s2 + w3_batch * s3
        x = x.permute(0, 2, 1)
        x = x.view(x.size(0), x.size(1), 128, -1)
        # set mask values
        x[:, :, 56: 72, 92: 97] = 0
        x[:, :, 33: 40, 33: 38] = 0
        x[:, :, 88: 95, 33: 38] = 0

        # 64X64 resolution
        x = self.main1(x)
        out_src_p1 = self.conv_binary1(x)
        out_src_p1 = out_src_p1.view(out_src_p1.size(0), -1)

        # 16X16 resolution
        x = self.main2(x)
        out_src_p2 = self.conv_binary2(x)
        out_src_p2 = out_src_p2.repeat(1, 1, 1, 16).view(out_src_p2.size(0), -1)
        # out_src_p2 = out_src_p2.view(out_src_p2.size(0), -1)

        #4X4 resolution
        x = self.main3(x)
        out_src_p3 = self.conv_binary3(x)
        out_src_p3 = out_src_p3.repeat(1, 1, 1, 256).view(out_src_p3.size(0), -1)
        # out_src_p3 = out_src_p3.view(out_src_p3.size(0), -1)
        out_src = torch.cat((out_src_p1, out_src_p2, out_src_p3), dim=1)
        # print(x.shape)
        out_cls = self.conv_cls(x)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))