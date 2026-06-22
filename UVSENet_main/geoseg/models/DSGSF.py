import torch
import torch.nn as nn
import torch.nn.functional as F
from .Cell import Spectral


# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.act = nn.Tanh()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.act(out)
        return out * x


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (1, 3, 5, 7)
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.act(out)
        return out * x


# 非局部模块
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.r = 16
        self.inter_channels = inter_channels or max(in_channels // 2, 1)

        self.g1 = nn.Conv2d(in_channels, self.inter_channels, 1, bias=False)
        self.g2 = nn.Conv2d(self.inter_channels, 1, 3, padding=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // self.r, 1),
            nn.LayerNorm([in_channels // self.r, 1, 1]),
            nn.PReLU(),
            nn.Conv2d(in_channels // self.r, in_channels, 1)
        )

    def spatial_pool(self, x):
        N, C, H, W = x.size()
        input_x = x.view(N, C, H * W).unsqueeze(1)  # [N,1,C,H*W]
        context_mask = self.g1(x)
        context_mask = self.g2(context_mask).view(N, 1, H * W)
        context_mask = self.softmax(context_mask).unsqueeze(-1)  # [N,1,H*W,1]
        context = torch.matmul(input_x, context_mask).view(N, C, 1, 1)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        channel_mul_term = torch.sigmoid(self.transform(context))
        return x * channel_mul_term


# 编码器
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=out_channels),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 7, padding=3, groups=out_channels),
            nn.Tanh()
        )
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_pool = self.pool(x)
        return x, x_pool


# 解码器
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 7, padding=3, groups=out_channels),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=out_channels),
            nn.Tanh()
        )
        self.spec_att = ChannelAttention(out_channels)
        self.spa_att = SpatialAttention(7)

    def forward(self, enc_feat, x):
        x = self.up(x)
        # pad if needed
        diff_h = enc_feat.shape[2] - x.shape[2]
        diff_w = enc_feat.shape[3] - x.shape[3]
        x = F.pad(x, (diff_w//2, diff_w - diff_w//2, diff_h//2, diff_h - diff_h//2))
        enc_feat = self.spec_att(enc_feat)
        x = torch.cat([enc_feat, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.spa_att(x)
        return x


# UNet 主干
class UNet(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(UNet, self).__init__()
        self.down1 = Encoder(in_ch, 32)
        self.down2 = Encoder(32, 64)
        self.mid_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.middle = NonLocalBlock(64)
        self.mid_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.up2 = Decoder(128, 64)
        self.up1 = Decoder(64, 32)
        self.last_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, num_classes, H, W]
        """
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x = self.mid_conv1(x)
        x = self.middle(x)
        x = self.mid_conv2(x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        x = self.last_conv(x)
        return x


# U_GC_LSTM 融合网络
class U_GC_LSTM(nn.Module):
    def __init__(self, bands, in_ch, num_classes):
        super(U_GC_LSTM, self).__init__()
        self.unet = UNet(in_ch, num_classes)
        self.mul_lstm = Spectral(bands, num_classes)

    def forward(self, x_img, x_spec):
        """
        x_img: [B, in_ch, H, W]
        x_spec: [B, bands, H, W]
        return: [B, num_classes, H, W]
        """
        B, _, H, W = x_img.shape
        # UNet 输出
        u_res = self.unet(x_img)  # [B, num_classes, H, W]

        # Spectral/LSTM 输出，先做 global pooling
        spec_pool = x_spec.view(B, x_spec.shape[1], -1).mean(dim=2)  # [B, bands]
        lstm_res = self.mul_lstm(spec_pool)  # [B, num_classes]
        lstm_res = lstm_res.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # 融合
        out = u_res + lstm_res
        return out
