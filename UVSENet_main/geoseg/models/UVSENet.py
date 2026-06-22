import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CMIA(nn.Module):

    def __init__(self, channels, ratio=16):
        super(CMIA, self).__init__()
        mid_channels = max(channels // ratio, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, 1, bias=False)
        )


        self.spatial_conv = nn.Conv2d(channels * 2, 2, kernel_size=3, padding=1, bias=False, groups=2)
        self.spatial_bn = nn.BatchNorm2d(2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)

        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        chn_out = avg_out + max_out
        chn_weights = F.softmax(chn_out, dim=1)

        spa_out = self.spatial_bn(self.spatial_conv(x))
        spa_weights = self.sigmoid(spa_out)

        weights = chn_weights * spa_weights

        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]

        fusion1 = x1 * (1 + w1)
        fusion2 = x2 * (1 + w2)

        fusion = torch.cat([fusion1, fusion2], dim=1)
        return fusion


class MACA(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, dilation_rates=[1, 3, 5]):
        super().__init__()

        self.branches = nn.ModuleList()
        for rate in dilation_rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=rate, dilation=rate, groups=in_channels, bias=False),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.weight_conv = nn.Conv2d(out_channels, len(dilation_rates), 1, bias=True)

        self.eca_conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        feats = [b(x) for b in self.branches]  # list of [B,C,H,W]
        U = sum(feats)

        # 自适应加权
        s = self.global_pool(U)  # [B,C,1,1]
        w = self.weight_conv(s)  # [B,num_scales,1,1]
        w = F.softmax(w, dim=1)

        V = 0
        for i, f in enumerate(feats):
            V += f * w[:, i:i + 1, :, :]

        # ECA 通道注意力
        ch = V.mean(dim=(2, 3), keepdim=False)  # [B,C]
        ch = ch.unsqueeze(1)
        ch = self.eca_conv(ch)  # [B,1,C]
        ch = ch.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        V = V * self.sigmoid(ch)

        return V


class SGMF(nn.Module):
    def __init__(self, channel, out_channels):
        super(SGMF, self).__init__()
        self.att = CMIA(channel, ratio=16)
        self.mf = MACA(channel, out_channels)
        self.conv = nn.Conv2d(channel * 2, out_channels, 1, bias=False)

    def forward(self, x1, x2):
        #         fusion=torch.cat([x1,x2],dim=1)
        # print('x',x.shape)
        fusion = self.att(x1, x2)
        # print('weights',weights.shape)

        out_features = self.conv(fusion)
        # print('fusion',fusion.shape)
        out_features = self.mf(out_features)
        # print('out_features',out_features.shape)

        return out_features


class SGCA(nn.Module):
    def __init__(self, C1, C2, out_channels, d_model=64, n_heads=8):
        super().__init__()
        self.d_model = d_model

        # -----------------------------
        # Stage 1: High-frequency Attention (Weight Only)
        # -----------------------------
        self.high_proj = nn.Linear(3 * C1, d_model)
        self.freq_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=False)

        self.attn_weight_proj = nn.Linear(d_model, 1)

        # -----------------------------
        # Stage 3: Cross-modal Attention
        # -----------------------------
        self.cross_q_proj = nn.Linear(C1, d_model)
        self.cross_k_proj = nn.Linear(C2, d_model)
        self.cross_v_proj = nn.Linear(C2, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=False)

        self.out_conv = nn.Conv2d(d_model, out_channels, 1)

        # DWT
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')
        self.idwt = DWTInverse(wave='haar', mode='zero')
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, embed):
        """
        x: (B, C1, H, W)
        embed: (B, C2, H, W)
        """
        B, C1, H, W = x.shape

        # =====================================================
        # Stage 1: DWT
        # =====================================================
        low, high = self.dwt(x)
        high = high[0]  # (B, C1, 3, H/2, W/2)

        # concat LH, HL, HH
        high_cat = torch.cat(
            [high[:, :, 0], high[:, :, 1], high[:, :, 2]],
            dim=1
        )  # (B, 3*C1, H/2, W/2)

        B, _, Hh, Wh = high_cat.shape
        N = Hh * Wh

        high_flat = high_cat.view(B, 3 * C1, -1).permute(2, 0, 1)  # (N, B, 3*C1)
        high_embed = self.high_proj(high_flat)  # (N, B, d_model)

        freq_attn_out, _ = self.freq_attn(
            high_embed, high_embed, high_embed
        )  # (N, B, d_model)

        # =====================================================
        # Stage 2: Attention → Spatial Weight
        # =====================================================
        attn_weight = self.attn_weight_proj(freq_attn_out)  # (N, B, 1)
        attn_weight = attn_weight.permute(1, 2, 0).view(B, 1, Hh, Wh)

        attn_weight = torch.sigmoid(attn_weight)
        attn_weight_h = F.interpolate(attn_weight, size=(H // 2, W // 2))


        alpha = torch.sigmoid(self.alpha)

        high = high * (1.0 + alpha * attn_weight_h.unsqueeze(2))

        x_guided = self.idwt((low, [high]))
        # =====================================================
        # Stage 3: Cross-modal Attention
        # =====================================================
        q = self.cross_q_proj(
            x_guided.flatten(2).permute(2, 0, 1)
        )  # (HW, B, d_model)

        k = self.cross_k_proj(
            embed.flatten(2).permute(2, 0, 1)
        )

        v = self.cross_v_proj(
            embed.flatten(2).permute(2, 0, 1)
        )

        cross_out, _ = self.cross_attn(q, k, v)
        cross_out = cross_out.permute(1, 2, 0).view(B, self.d_model, H, W)

        out = self.out_conv(cross_out)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([skip, x], dim=1)

        x = self.conv(x)
        return x


class enUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(enUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        # print('x1',x1.shape)
        x2 = self.down1(x1)
        # print('x2', x2.shape)
        x3 = self.down2(x2)
        # print('x3', x3.shape)
        x4 = self.down3(x3)
        # print('x4', x4.shape)
        x5 = self.down4(x4)
        # print('x5', x5.shape)
        return x1, x2, x3, x4, x5

class UVNet(nn.Module):
    def __init__(self, n_channels=64, n_classes=2, bilinear=True):
        super(UVNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.enUNet1 = enUNet(3, n_classes)
        self.enUNet2 = enUNet(64, n_classes)

        self.up1 = UNetUpBlock(512, 512, 512)
        self.up2 = UNetUpBlock(512, 256, 256)
        self.up3 = UNetUpBlock(256, 128, 128)
        self.up4 = UNetUpBlock(128, 64, 64)
        self.outc = OutConv(64, n_classes)

        self.wmmf1 = SGMF(64, 64)
        self.wmmf2 = SGMF(128, 128)
        self.wmmf3 = SGMF(256, 256)
        self.wmmf4 = SGMF(512, 512)
        self.wmmf5 = SGCA(C1=512, C2=512, out_channels=512)

    def forward(self, x, embed):
        ## Annual images
        x1, x2, x3, x4, x5 = self.enUNet1(x)
        ## Seasonal images
        x11, x21, x31, x41, x51 = self.enUNet2(embed)

        x_1 = self.wmmf1(x1, x11)
        x_2 = self.wmmf2(x2, x21)
        x_3 = self.wmmf3(x3, x31)
        x_4 = self.wmmf4(x4, x41)
        x_5 = self.wmmf5(x5, x51)

        x4 = self.up1(x_5, x_4)
        x3 = self.up2(x4, x_3)
        x2 = self.up3(x3, x_2)
        x1 = self.up4(x2, x_1)
        logits = self.outc(x1)

        return logits

