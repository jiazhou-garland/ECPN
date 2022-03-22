import torch
import timm
import torch.nn as nn
from utils.efficientnet_utils import MemoryEfficientSwish, Swish, Conv2dStaticSamePadding

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = self.swish(x)
        return x

class UpConvBlock_Transpose(nn.Module):
    def __init__(self, in_features, up_scale, padding=1):
        super(UpConvBlock_Transpose, self).__init__()

        self.conv_for_transposed = nn.Conv2d(in_features, out_channels=1, kernel_size=1)
        self.transposed_conv = nn.ConvTranspose2d(
            1, 1, kernel_size=2 * up_scale, stride=up_scale, padding=padding, bias=False)
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        x = self.conv_for_transposed(x)
        x = self.swish(x)
        x = self.transposed_conv(x)
        return x

class UpConvBlock_subpixel(nn.Module):
    def __init__(self, in_features, up_scale, out_channels):
        super(UpConvBlock_subpixel, self).__init__()
        self.conv_for_subpixel = nn.Conv2d(in_features, out_channels=out_channels, kernel_size=1)
        self.subpixel_conv = nn.PixelShuffle(up_scale)
        self.swish = MemoryEfficientSwish()

    def forward(self, x):
        x = self.conv_for_subpixel(x)
        x = self.swish(x)
        x = self.subpixel_conv(x)
        return x

class Bidecoder(nn.Module):
    def __init__(self, onnx_export=False):
        super(Bidecoder, self).__init__()
        # Conv layers
        self.conv6_up = SeparableConvBlock(160, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(64, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(48, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(24, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(48, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(64, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(160, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(272, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.ConvTranspose2d(272, 160, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.p5_upsample = nn.ConvTranspose2d(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.p4_upsample = nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.p3_upsample = nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.p4_downsample = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)
        self.p5_downsample = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)
        self.p6_downsample = nn.Conv2d(64, 160, kernel_size=3, stride=2, padding=1)
        self.p7_downsample = nn.Conv2d(160, 272, kernel_size=3, stride=2, padding=1)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        """
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

class Down_to_Up_decoder(nn.Module):
    def __init__(self, onnx_export=False):
        super(Down_to_Up_decoder, self).__init__()
        # Conv layers
        self.conv6_up = SeparableConvBlock(160, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(64, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(48, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(24, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(48, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(64, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(160, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(272, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.ConvTranspose2d(272, 160, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.p5_upsample = nn.ConvTranspose2d(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.p4_upsample = nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.p3_upsample = nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        """
            P7_0 -------------------------> P7_2 -------->
               |-------------|
                             ↓
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑
                             ↓
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑
                             ↓
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑
                             |--------------↓
            P3_0 -------------------------> P3_2 -------->
        """

        p3_0, p4_0, p5_0, p6_0, p7_0 = inputs

        p6_1 = self.conv6_up(self.swish(p6_0 + self.p6_upsample(p7_0)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_1 = self.conv5_up(self.swish(p5_0 + self.p5_upsample(p6_1)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_1 = self.conv4_up(self.swish(p4_0+ self.p4_upsample(p5_1)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_2 = self.conv3_up(self.swish(p3_0 + self.p3_upsample(p4_1)))

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_2 = self.conv4_down(self.swish(p4_0 + p4_1))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_2 = self.conv5_down(self.swish(p5_0 + p5_1))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_2 = self.conv6_down(self.swish(p6_0 + p6_1))

        # Connections for P7_0 and P6_2 to P7_2
        p7_2 = self.conv7_down(self.swish(p7_0))

        return p3_2, p4_2, p5_2, p6_2, p7_2

class Up_to_Down_decoder(nn.Module):
    def __init__(self, onnx_export=False):
        super(Up_to_Down_decoder, self).__init__()
        # Conv layers
        self.conv6_up = SeparableConvBlock(160, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(64, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(48, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(24, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(48, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(64, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(160, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(272, onnx_export=onnx_export)

        self.p4_downsample = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)
        self.p5_downsample = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)
        self.p6_downsample = nn.Conv2d(64, 160, kernel_size=3, stride=2, padding=1)
        self.p7_downsample = nn.Conv2d(160, 272, kernel_size=3, stride=2, padding=1)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        """
            P7_0 -------------------------> P7_2 -------->
               |                              ↑
                                              |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                                              |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                                              |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                                              |
            P3_0 -------------------------> P3_2 -------->
        """

        p3_0, p4_0, p5_0, p6_0, p7_0 = inputs

        p6_1 = self.conv6_up(self.swish(p6_0))

        p5_1 = self.conv5_up(self.swish(p5_0))

        p4_1 = self.conv4_up(self.swish(p4_0))

        p3_2 = self.conv3_up(self.swish(p3_0))

        p4_2 = self.conv4_down(self.swish(p4_0 + p4_1 + self.p4_downsample(p3_2)))

        p5_2 = self.conv5_down(self.swish(p5_0 + p5_1 + self.p5_downsample(p4_2)))

        p6_2 = self.conv6_down(self.swish(p6_0 + p6_1 + self.p6_downsample(p5_2)))

        p7_2 = self.conv7_down(self.swish(p7_0 + self.p7_downsample(p6_2)))

        return p3_2, p4_2, p5_2, p6_2, p7_2

class General_decoder(nn.Module):
    def __init__(self, onnx_export=False):
        super(General_decoder, self).__init__()
        # Conv layers
        self.conv6_up = SeparableConvBlock(160, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(64, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(48, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(24, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(48, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(64, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(160, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(272, onnx_export=onnx_export)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-----------------------------↑
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-----------------------------↑
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-----------------------------↑
            P3_0 -------------------------> P3_2 -------->
        """

        p3_0, p4_0, p5_0, p6_0, p7_0 = inputs

        p6_1 = self.conv6_up(p6_0)
        p5_1 = self.conv5_up(p5_0)
        p4_1 = self.conv4_up(p4_0)

        p3_2 = p3_0
        p4_2 = self.conv4_down(self.swish(p4_0 + p4_1))
        p5_2 = self.conv5_down(self.swish(p5_0 + p5_1))
        p6_2 = self.conv6_down(self.swish(p6_0 + p6_1))
        p7_2 = p7_0

        return p3_2, p4_2, p5_2, p6_2, p7_2

class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, kernel=1, use_bs=True, padding=1):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, bias=False, padding=padding)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MLEEP(nn.Module):

    def __init__(self, multi_inputs=False, upsample_type='Transpose-conv', decoder_type='Bidecoder'):
        super(MLEEP, self).__init__()
        model = timm.create_model('efficientnetv2_rw_s', features_only=True, pretrained=True)
        self.model = model
        self.Bidecoder = Bidecoder(onnx_export=False)
        self.General_decoder = General_decoder(onnx_export=False)
        self.Down_to_up_decoder = Down_to_Up_decoder(onnx_export=False)
        self.Up_to_Down_decoder = Up_to_Down_decoder(onnx_export=False)
        if upsample_type == 'Transpose-conv':
            self.up_block_1 = UpConvBlock_Transpose(24, 2, padding=1)
            self.up_block_2 = UpConvBlock_Transpose(48, 4, padding=2)
            self.up_block_3 = UpConvBlock_Transpose(64, 8, padding=4)
            self.up_block_4 = UpConvBlock_Transpose(160, 16, padding=8)
            self.up_block_5 = UpConvBlock_Transpose(272, 32, padding=16)
        elif upsample_type == 'Sub-pixel-conv':
            self.up_block_1 = UpConvBlock_subpixel(24, up_scale=2, out_channels=4)
            self.up_block_2 = UpConvBlock_subpixel(48, up_scale=4, out_channels=16)
            self.up_block_3 = UpConvBlock_subpixel(64, up_scale=8, out_channels=64)
            self.up_block_4 = UpConvBlock_subpixel(160, up_scale=16, out_channels=256)
            self.up_block_5 = UpConvBlock_subpixel(272, up_scale=32, out_channels=1024)
        self.block_cat = SingleConvBlock(5, 1, stride=1, use_bs=False, padding=0) # hed fusion method
        self.multi_inputs = multi_inputs
        self.c_conv = SingleConvBlock(7, 28, stride=1, kernel=3, use_bs=True)
        self.SE_block = SELayer(channel=28, reduction=14)
        self.c_fusion = SingleConvBlock(28, 1, stride=1, kernel=3, use_bs=True)
        self.decoder_type = decoder_type

    def forward(self, x):

        if self.multi_inputs:
            input_list = []
            for i in range(3):
                input = (x[:, 0+i, :, :], x[:, 3+i, :, :], x[:, 6+i, :, :], x[:, 9+i, :, :], x[:, 12+i, :, :],
                        x[:, 15+i, :, :], x[:, 18+i, :, :])
                input = torch.stack(input, dim=1)
                input = self.c_conv(input)# Bx7xHxW
                input = self.SE_block(input)# Bx28xHxW
                input = self.c_fusion(input)# Bx1xHxW
                input_list.append(input)
            x = torch.cat(input_list, dim=1)# Bx3xHxW
        output = self.model(x)

        features = (output[0], output[1], output[2], output[3], output[4])

        if self.decoder_type == 'Bidecoder':
            p3_out, p4_out, p5_out, p6_out, p7_out = self.Bidecoder(features)
        elif self.decoder_type == 'General_decoder':
            p3_out, p4_out, p5_out, p6_out, p7_out = self.General_decoder(features)
        elif self.decoder_type == 'Down_to_Up_decoder':
            p3_out, p4_out, p5_out, p6_out, p7_out = self.Down_to_up_decoder(features)
        elif self.decoder_type == 'Up_to_Down_decoder':
            p3_out, p4_out, p5_out, p6_out, p7_out = self.Up_to_Down_decoder(features)

        out_1 = self.up_block_1(p3_out)
        out_2 = self.up_block_2(p4_out)
        out_3 = self.up_block_3(p5_out)
        out_4 = self.up_block_4(p6_out)
        out_5 = self.up_block_5(p7_out)
        results = [out_1, out_2, out_3, out_4, out_5]

        # concatenate multiscale outputs and fuse them
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW
        results.append(block_cat)

        return results

if __name__ == '__main__':

    inputs = torch.rand(2, 3, 512, 512)
    model = MLEEP(multi_inputs=False, upsample_type='Transpose-conv', decoder_type='Bidecoder')
    # print(model)
    o = model(inputs)
    for i in range(6):
        print(o[i].size())
