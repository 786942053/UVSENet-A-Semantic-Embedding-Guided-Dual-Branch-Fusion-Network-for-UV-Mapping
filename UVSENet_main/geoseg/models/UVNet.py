import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Branch_Attention(nn.Module):
    def __init__(self,channel,ratio=16):
        super(Branch_Attention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        # replace fc with 1x1 Conv
        self.fc1=nn.Conv2d(channel,channel//ratio,1,bias=False)
        self.relu=nn.ReLU()
        self.fc2=nn.Conv2d(channel//ratio,2,1,bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x1,x2):
        x = torch.cat([x1, x2], dim=1)
        avg_out=self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out=self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out=avg_out+max_out
        weights=self.sigmoid(out)

        fusion1 = x1 + x1 * weights[:, 0, :, :].unsqueeze(1)
        fusion2 = x2 + x2 * weights[:, 1, :, :].unsqueeze(1)
        fusion = torch.cat([fusion1, fusion2], dim=1)


        return fusion

class Multiscale_Feature_Module(nn.Module):
    def __init__(self,out_channels):
        super(Multiscale_Feature_Module, self).__init__()


        self.conv0_1=nn.Conv2d(out_channels,out_channels,(1,3),padding=(0,1),groups=out_channels)
        self.conv0_2=nn.Conv2d(out_channels,out_channels,(3,1),padding=(1,0),groups=out_channels)

        self.conv1_1=nn.Conv2d(out_channels,out_channels,(1,7),padding=(0,3),groups=out_channels)
        self.conv1_2=nn.Conv2d(out_channels,out_channels,(7,1),padding=(3,0),groups=out_channels)

        self.conv2_1=nn.Conv2d(out_channels,out_channels,(1,11),padding=(0,5),groups=out_channels)
        self.conv2_2=nn.Conv2d(out_channels,out_channels,(11,1),padding=(5,0),groups=out_channels)

        self.conv3=nn.Conv2d(out_channels,out_channels,1)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):


#         print(x.shape)
        u=x.clone()
        attn_0=self.conv0_1(x)
        attn_0=self.conv0_2(attn_0)

        attn_1=self.conv1_1(x)
        attn_1=self.conv1_2(attn_1)

        attn_2=self.conv2_1(x)
        attn_2=self.conv2_2(attn_2)

        attn=x+attn_0+attn_1+attn_2
        attn=self.conv3(attn)
        out=attn*u

        return out
class WMFFM  (nn.Module):
    def __init__(self, channel,out_channels):
        super(WMFFM , self).__init__()
        self.att=Branch_Attention( channel,ratio=16)
        self.mf=Multiscale_Feature_Module(out_channels)

        self.conv = nn.Conv2d(channel, out_channels, 1, bias=False)
    def forward(self, x1,x2):
        # x=torch.cat([x1,x2],dim=1)
        # print('x',x.shape)
        fusion=self.att(x1,x2)
        # print('weights',weights.shape)


        fusion=self.conv (fusion)
        # print('fusion',fusion.shape)
        out_features = self.mf(fusion)
        # print('out_features',out_features.shape)

        return out_features




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
        return x1,x2,x3,x4,x5


class deUNet(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(deUNet, self).__init__()

        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x1,x2,x3,x4,x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

class UVNet(nn.Module):
    def __init__(self, n_channels=64, n_classes=2, bilinear=True):
        super(UVNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.enUNet1=enUNet(3, n_classes)
        self.enUNet2 = enUNet(64, n_classes)
        self.deUNet=deUNet(2)

        self.wmmf1=WMFFM (64*2,64)
        self.wmmf2 = WMFFM (128*2, 128)
        self.wmmf3 = WMFFM (256*2 , 256)
        self.wmmf4 = WMFFM (512*2 , 512)
        self.wmmf5 = WMFFM (512*2, 512)

    def forward(self, x, embed):
        ## Annual images
        x1, x2, x3, x4, x5=self.enUNet1(x)
        ## Seasonal images
        x11, x21, x31, x41, x51 = self.enUNet2(embed)

        x_1=self.wmmf1(x1,x11)
        x_2 = self.wmmf2(x2, x21)
        x_3 = self.wmmf3(x3, x31)
        x_4 = self.wmmf4(x4, x41)
        x_5 = self.wmmf5(x5, x51)

        logits = self.deUNet(x_1,x_2,x_3,x_4,x_5)

        return logits

