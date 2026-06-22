import torch
from torch import nn


class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DRFB_module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DRFB_module, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    def __init__(self, channel, n_class):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, n_class, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        """
        Facilitating feature fusion across multiple scales and incorporating cross-layer information interaction
        """
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


class OneBranch(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(OneBranch, self).__init__()

        n1 = 64
        channel = 32
        n_class = 256
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])

        self.drfb2_1 = DRFB_module(256, channel)
        self.drfb3_1 = DRFB_module(512, channel)
        self.drfb4_1 = DRFB_module(1024, channel)
        self.agg = aggregation(channel, n_class)

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        """
        Encoding stage
        """
        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool(x0_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))

        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))

        x4_0 = self.conv4_0(self.pool(x3_0))

        x2_0_rfb = self.drfb2_1(x2_0)  # channel -> 32

        x3_0_rfb = self.drfb3_1(x3_0)  # channel -> 32

        x4_0_rfb = self.drfb4_1(x4_0)  # channel -> 32
        """
        Decoding stage
        """
        x2_0_0 = self.agg(x4_0_rfb, x3_0_rfb, x2_0_rfb)

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_0_0)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        output = self.final(x0_3)
        return output


# class SBHE(nn.Module):
#     def __init__(self, in_channels2=3, out_channels2=2, in_channels1=64, out_channels1=2):
#         super(SBHE, self).__init__()
#         self.num_modules = 1
#         self.channels = [256, 512, 1024]
#         self.OneBranchs2 = OneBranch(in_ch=in_channels2, out_ch=out_channels2)
#         self.OneBranch = OneBranch(in_ch=in_channels1, out_ch=out_channels1)

#     def forward(self, inputs2, inputs1):
#         outputs2 = self.OneBranchs2(inputs2)
#         '''
#         Enhancement of Sentinel-2 Data for Improved Information Extraction
#         '''
#         predfoot = torch.sigmoid(outputs2)
#         result, indices = torch.max(predfoot, dim=1)
#         indices = torch.unsqueeze(indices, dim=1)
#         '''
#         Enhancing Building Footprint Data for Accurate Height Estimation
#         '''
#         outputs1 = self.OneBranch(inputs1)
#         final = outputs1 * indices
#         return final
class SBHE(nn.Module):
    def __init__(self, in_channels2=3, out_channels2=2, in_channels1=64, out_channels1=2):
        super(SBHE, self).__init__()
        self.channels = [256, 512, 1024]
        self.OneBranchs2 = OneBranch(in_ch=in_channels2, out_ch=out_channels2)
        self.OneBranch = OneBranch(in_ch=in_channels1, out_ch=out_channels1)
        self.fusion_conv = nn.Conv2d(out_channels1 + out_channels2, out_channels2, kernel_size=1)

    def forward(self, inputs2, inputs1):
        out2 = self.OneBranchs2(inputs2)  # [B, out_channels2, H, W]
        out1 = self.OneBranch(inputs1)    # [B, out_channels1, H, W]

        # 拼接
        fused = torch.cat([out1, out2], dim=1)  # channel 维度拼接

        # 1x1卷积融合成最终输出
        final = self.fusion_conv(fused)  # [B, out_channels2, H, W]

        return final  # logits

if __name__ == '__main__':
    '''
    SBHE Model Runs
    '''
    images2 = torch.randn(1, 4, 384, 384)
    images1 = torch.randn(1, 2, 384, 384)
    model = SBHE(in_channels2=4, out_channels2=2, in_channels1=2, out_channels1=1)
    footprint, height = model(images2, images1)
    print(footprint.shape, height.shape)