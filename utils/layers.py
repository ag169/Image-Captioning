import torch
import torch.nn as nn
import math
from utils.gen_ops import lowercase


class ConvBnRelu(nn.Module):
    r"""
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple, optional): Size of the convolving kernel. Default: 3
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        pad (int or tuple or str, optional): Zero-padding added to both sides of the input. Default: 'same'
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        relu (bool, optional): If ``True``, adds a ReLU to the sequential block
    """
    def __init__(self, in_channels, out_channels, pad='same', kernel_size=3, stride=1, dilation=1, groups=1,
                 bias=True, relu=True, bn=True):
        super(ConvBnRelu, self).__init__()

        if type(pad) == str:
            pad = lowercase(pad)
            if pad == 'valid':
                padding = 0
            elif pad == 'same':
                if type(kernel_size) == int:
                    kernel_size = (kernel_size, kernel_size)
                if type(stride) == int:
                    stride = (stride, stride)
                if type(dilation) == int:
                    dilation = (dilation, dilation)

                padding = [0, 0]
                for i in range(2):
                    padding[i] = math.ceil((dilation[i]*(kernel_size[i] - 1) - stride[i] + 1) / 2)

                padding = tuple(padding)
            else:
                print('Invalid padding mode')
                raise NotImplementedError
        elif type(pad) == int or type(pad) == tuple:
            padding = pad
        else:
            print('Invalid type of pad')
            raise TypeError

        block_list = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                bias=bias and not bn)]

        if bn:
            block_list.append(nn.BatchNorm2d(out_channels))

        if relu:
            block_list.append(nn.ReLU6())

        self.block = nn.Sequential(*block_list)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x)


class DeconvBnRelu(nn.Module):
    r"""
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple, optional): Size of the convolving kernel. Default: 3
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            pad (int or tuple or str, optional): Zero-padding added to both sides of the input. Default: 'same'
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
            relu (bool, optional): If ``True``, adds a ReLU to the sequential block
        """
    def __init__(self, in_channels, out_channels, pad='same', kernel_size=3, stride=1, dilation=1, groups=1, bias=True,
                 relu=True, bn=True):
        super(DeconvBnRelu, self).__init__()

        if type(pad) == str:
            pad = lowercase(pad)
            if pad == 'valid':
                padding = 0
            elif pad == 'same':
                if type(kernel_size) == int:
                    kernel_size = (kernel_size, kernel_size)
                if type(stride) == int:
                    stride = (stride, stride)
                if type(dilation) == int:
                    dilation = (dilation, dilation)

                padding = [0, 0]
                for i in range(2):
                    padding[i] = math.ceil((dilation[i] * (kernel_size[i] - 1) + 1 - stride[0]) / 2)

                padding = tuple(padding)
            else:
                print('Invalid padding mode')
                raise NotImplementedError
        elif type(pad) == int or type(pad) == tuple:
            padding = pad
        else:
            print('Invalid type of pad')
            raise TypeError

        block_list = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation,
                                         groups=groups, bias=bias and not bn)]

        if bn:
            block_list.append(nn.BatchNorm2d(out_channels))

        if relu:
            block_list.append(nn.ReLU6())

        self.block = nn.Sequential(*block_list)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=(3, 6, 9)):
        super(ASPP, self).__init__()

        self.convs = nn.ModuleList()

        self.convs.append(ConvBnRelu(in_channels, in_channels, kernel_size=3))

        for di in dilation:
            self.convs.append(ConvBnRelu(in_channels, in_channels, kernel_size=kernel_size, dilation=di))

        self.pool = nn.MaxPool2d(5, 1, 2)

        self.conv_out = ConvBnRelu(in_channels * (2 + len(dilation)), out_channels, kernel_size=3)

    def forward(self, x):
        t = []

        for i in range(len(self.convs)):
            t.append(self.convs[i](x))

        t.append(self.pool(x))
        out = torch.cat(t, 1)
        return self.conv_out(out)


if __name__ == '__main__':
    in_c = 24
    out_c = 36
    ks = 4
    s = 2
    d = 1
    p = 'same'
    # p = 0

    # block = ConvBnRelu(in_c, out_c, pad=p, kernel_size=ks, stride=s, dilation=d)
    # block = DeconvBnRelu(in_c, out_c, pad=p, kernel_size=ks, stride=s, dilation=d)
    block = ASPP(in_c, out_c, dilation=(6, 12, 18))

    w = 256
    h = 384
    ip = torch.rand(size=[2, in_c, h, w])

    print(ip.size())
    op = block(ip)
    print(op.size())
