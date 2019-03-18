import torch
from torch import nn
from architectures.shared import networks
from functools import partial

def get_network():
    n_channels = 3
    ngf = 64
    norm_layer = partial(nn.InstanceNorm2d, affine=True)
    gen = networks.ResnetEncoderDecoder(input_nc=n_channels,
                                        output_nc=n_channels,
                                        ngf=ngf,
                                        n_blocks=4,
                                        n_downsampling=4,
                                        norm_layer=norm_layer)
    return gen

if __name__ == '__main__':
    print("ooo")
