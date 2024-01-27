import torch
import torch.nn.functional as F
from torch import nn, Tensor
from thop import profile


# 空洞卷积+自建特征融合模块+修改MSCA
class DropBlock(nn.Module):
    def __init__(self, block_size: int = 5, p: float = 0.1):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


# 定义Atrous Separable Convolution层
class AtrousSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(AtrousSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding=dilation, dilation=dilation,
                                        groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# 定义Separable Convolution层
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels,
                                        padding=padding)

        # 逐点卷积层
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class Conv1(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(Conv1, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,dilation=1),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class Conv2(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(Conv2, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2,dilation=2),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class Conv3(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(Conv3, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3,dilation=3),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class NFEM(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(NFEM, self).__init__()
        self.dconv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=(1, 1))
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), dilation=1,padding=(0, 1)),  # 3 18
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), dilation=1,padding=(1, 0))  # 18 6
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), dilation=2, padding=(0, 2)),  # 3 18
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), dilation=2, padding=(2, 0))  # 18 6
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), dilation=3, padding=(0, 3)),  # 3 18
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), dilation=3, padding=(3, 0))  # 18 6
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1)  # 3 18
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, bias=False, kernel_size=1),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.dconv(x)
        x2 = self.branch1(x1)
        x3 = self.branch2(x1)
        x4 = self.branch3(x1)
        x5 = self.branch4(x1)
        x6 = x2 + x3 + x4 + x5
        x7 = self.conv(x6)
        x8 = x7 * x1
        return x8



class Conv1x1(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(Conv1x1, self).__init__(
            nn.Conv2d(in_channels, num_classes, bias=False, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, bias=False, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.Sigmoid()
        )


class Net(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_c: int = 32):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.NFEM = NFEM(in_channels, base_c)

        self.conv1 = Conv1(in_channels, base_c)
        self.conv2 = Conv2(base_c, base_c)
        self.conv3 = Conv3(base_c, base_c)
        self.conv1_1 = Conv1(base_c, base_c)
        self.down1 = nn.MaxPool2d(2, stride=2)

        self.conv1_2 = Conv1(base_c, base_c * 2)
        self.conv11_2 = Conv1(base_c * 2, base_c * 2)
        self.conv2_2 = Conv2(base_c * 2, base_c * 2)
        self.conv3_2 = Conv3(base_c * 2, base_c * 2)
        self.down2 = nn.MaxPool2d(2, stride=2)

        self.conv1_3 = Conv1(base_c * 2, base_c * 4)
        self.conv11_3 = Conv1(base_c * 4, base_c * 4)
        self.conv2_3 = Conv2(base_c * 4, base_c * 4)
        self.conv3_3 = Conv3(base_c * 4, base_c * 4)
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv4 = Conv1x1(base_c * 4, base_c)
        self.conv5 = Conv1x1(base_c * 2, base_c)

        self.conv6 = Conv1(base_c * 4, base_c * 4)
        self.conv7 = Conv2(base_c * 4, base_c * 4)
        self.conv8 = Conv3(base_c * 4, base_c)
        self.conv9 = Conv1x1(base_c * 4, base_c * 4)

        self.outconv = OutConv(base_c, num_classes)

    def forward(self, x):
        x_1 = self.NFEM(x)

        # first layer
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv2(x3)
        x5 = self.conv1_1(x4)
        x6 = self.conv3(x5)
        # second layer
        x1_1 = self.down1(x1)
        x8 = self.conv1_2(x1_1)
        x9 = self.conv2_2(x8)
        x10 = self.conv3_2(x9)
        x11 = self.conv2_2(x10)
        x12 = self.conv11_2(x11)
        x13 = self.conv3_2(x12)
        # three layer
        x8_8 = self.down2(x8)
        x14 = self.conv1_3(x8_8)
        x15 = self.conv2_3(x14)
        x16 = self.conv3_3(x15)
        x17 = self.conv2_3(x16)
        x18 = self.conv11_3(x17)
        x19 = self.conv3_3(x18)

        x20 = self.up1(x19)
        x21 = self.conv4(x20)
        x22 = self.up2(x13)
        x24 = self.conv5(x22)

        # multi scale
        x25 = torch.cat([x_1, x6, x21, x24], dim=1)
        x29 = self.conv9(x25)
        x26 = self.conv6(x25+x29)
        x27 = self.conv7(x25+x26)
        x28 = self.conv8(x25+x27)


        #x30 = x26 + x27 + x28 + x29
        outputs = self.outconv(x28)

        return outputs


if __name__ == "__main__":
    model = Net()
    input = torch.randn(4, 3, 256, 256)  # .to(device)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
