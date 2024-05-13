# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:33:46 2024

@author: loulo
"""
from nnunet.network_architecture.custom_modules.helperModules import Identity
from nnunet.network_architecture.custom_modules.conv_blocks import ConvDropoutNormReLU

import torch
import torch.nn as nn
from .cbam import CBAM
from copy import deepcopy    
import numpy as np    

class StackedConvCBAMLayers(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, network_props, cbam_props, num_convs, first_stride=None, cbam_ratio=16):
        """
        Stacked convolutional layers followed by CBAM.
        :param input_channels: Number of input channels.
        :param output_channels: Number of output channels, outputted by the stacked conv and given to CBAM
        :param kernel_size: Kernel size for convolution.
        :param network_props: Network properties dict containing conv_op, norm_op, etc.
        :param num_convs: Number of convolutional layers.
        :param first_stride: Stride for the first convolutional layer.
        :param cbam_ratio: Reduction ratio for CBAM.
        """
        super(StackedConvCBAMLayers, self).__init__()

        network_props = deepcopy(network_props)      # network_props is a dict and mutable, so we deepcopy to be safe.
        network_props_first = deepcopy(network_props)
        cbam_props = deepcopy(cbam_props)

        if first_stride is not None:
            network_props_first['conv_op_kwargs']['stride'] = first_stride

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(input_channels, output_channels, kernel_size, network_props_first),
            *[ConvDropoutNormReLU(output_channels, output_channels, kernel_size, network_props) for _ in 
              range(num_convs - 1)],
            CBAM(output_channels, ratio=cbam_ratio, props=cbam_props)
        )

    def forward(self, x):
        return self.convs(x)
