#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch

##import du decodeur
from nnunet.network_architecture.generic_modular_UNet import PlainConvUNetDecoder
##import de get default config pour les props
from nnunet.network_architecture.generic_modular_UNet import get_default_network_config

##on import les StackedConvCbamLayers au lieu des StackedConvLayers

from nnunet.network_architecture.custom_modules.conv_blocks import StackedConvLayers
from nnunet.network_architecture.custom_modules.cbam_conv_blocks import StackedConvCBAMLayers

from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
import numpy as np
from torch.optim import SGD

"""
The idea behind this modular U-net is that we decouple encoder and decoder and thus make things a) a lot more easy to 
combine and b) enable easy swapping between segmentation or classification mode of the same architecture
"""

## faire get_default_network_cbam_config qui contient que les props du Cbam, et donner les deux a la fonction plainconvcbamUnet et au codeur 
##le decodeur ne prend que get default network_config

## a modifier!!
def get_default_network_CBAM_config(dim=2):
    """
    returns a dictionary that contains pointers to conv, nonlin and norm ops and the default kwargs I like to use
    :return:
    """
    ##the standard order of dimensions is typically [NxCxHxW] for 2D inputs and [NxCxDxHxW] for 3D inputs

    props = {}

    ##il faut flatten pour donner au MLP
    ##**********************CHANNEL ATTENTION MODULE VERSION LOUIS***************************
    if dim == 2:
        spatial_dims = (2,3)
    elif dim == 3:
        spatial_dims = (2,3,4)
    props['channel_mean_op_kwargs'] =  {'dim': spatial_dims,'keepdim':False}
    props['channel_max_op_kwargs'] =  {'dim': spatial_dims,'keepdim':False}
    if dim == 2:
        props['channel_resize_op'] = lambda x: x.view(x.size(0),x.size(1),1,1)
    elif dim == 3:
        props['channel_resize_op'] = lambda x: x.view(x.size(0),x.size(1),1,1,1)
    else:
        raise NotImplementedError
    ##**********************FIN CHANNEL ATTENTION MODULE VERSION LOUIS***************************




    ##**********************SPATIAL ATTENTION MODULE***************************
    if dim == 2:
        props['conv_op'] = nn.Conv2d
    elif dim == 3:
        props['conv_op'] = nn.Conv3d
    else:
        raise NotImplementedError
    props['spatial_mean_op_kwargs'] =  {'dim': 1, 'keepdim': True}
    props['spatial_max_op_kwargs'] =  {'dim': 1, 'keepdim': True}
    ##**********************FINSPATIAL ATTENTION MODULE***************************

    return props





##AS IS PlainConvUNetEncoder, but the current stage line is replaced with the StackedConvCbamLayers instead of StackedConvLayers
class PlainConvUNetCBAMEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, cbam_props, default_return_skips=True,
                 max_num_features=480):
        """
        Following UNet building blocks can be added by utilizing the properties this class exposes (TODO)

        this one includes the bottleneck layer!

        :param input_channels:
        :param base_num_features:
        :param num_blocks_per_stage:
        :param feat_map_mul_on_downscale:
        :param pool_op_kernel_sizes:
        :param conv_kernel_sizes:
        :param props:
        """
        super(PlainConvUNetCBAMEncoder, self).__init__()

        self.default_return_skips = default_return_skips
        self.props = props

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes)

        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages

        self.num_blocks_per_stage = num_blocks_per_stage  # decoder may need this

        current_input_features = input_channels
        for stage in range(num_stages):
            current_output_features = min(int(base_num_features * feat_map_mul_on_downscale ** stage), max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]

            ##ON NE CHANGE QUE CA ! ON UTILISE LES STACKED CONV CBAM LAYERS!
            current_stage = StackedConvCBAMLayers(current_input_features, current_output_features, current_kernel_size,
                                              props, cbam_props, num_blocks_per_stage[stage], current_pool_kernel_size)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)
        self.output_features = current_output_features

    def forward(self, x, return_skips=None):
        """

        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        for s in self.stages:
            x = s(x)
            if self.default_return_skips:
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        if return_skips:
            return skips
        else:
            return x

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, pool_op_kernel_sizes, num_blocks_per_stage_encoder,
                                        feat_map_mul_on_downscale, batch_size):
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)

        tmp = num_blocks_per_stage_encoder[0] * np.prod(current_shape) * base_num_features \
              + num_modalities * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool + 1):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_blocks_per_stage_encoder[p]
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat
        return tmp * batch_size


##le decodeur ne change pas, on l'importe de generic modular Unet


class PlainConvCBAMUNet(SegmentationNetwork):
    use_this_for_batch_size_computation_2D = 1167982592.0
    use_this_for_batch_size_computation_3D = 1152286720.0

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, cbam_props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None):
        super(PlainConvCBAMUNet, self).__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = PlainConvUNetCBAMEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                            feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                            props, cbam_props, default_return_skips=True, max_num_features=max_features)
        self.decoder = PlainConvUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props,
                                            deep_supervision, upscale_logits)
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, num_blocks_per_stage_encoder,
                                        num_blocks_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        enc = PlainConvUNetCBAMEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                   num_modalities, pool_op_kernel_sizes,
                                                                   num_blocks_per_stage_encoder,
                                                                   feat_map_mul_on_downscale, batch_size)
        dec = PlainConvUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                   num_classes, pool_op_kernel_sizes,
                                                                   num_blocks_per_stage_decoder,
                                                                   feat_map_mul_on_downscale, batch_size)

        return enc + dec

    @staticmethod
    def compute_reference_for_vram_consumption_3d():
        patch_size = (160, 128, 128)
        pool_op_kernel_sizes = ((1, 1, 1),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2))
        conv_per_stage_encoder = (2, 2, 2, 2, 2, 2)
        conv_per_stage_decoder = (2, 2, 2, 2, 2)

        return PlainConvCBAMUNet.compute_approx_vram_consumption(patch_size, 32, 512, 4, 3, pool_op_kernel_sizes,
                                                             conv_per_stage_encoder, conv_per_stage_decoder, 2, 2)

    @staticmethod
    def compute_reference_for_vram_consumption_2d():
        patch_size = (256, 256)
        pool_op_kernel_sizes = (
            (1, 1), # (256, 256)
            (2, 2), # (128, 128)
            (2, 2), # (64, 64)
            (2, 2), # (32, 32)
            (2, 2), # (16, 16)
            (2, 2), # (8, 8)
            (2, 2)  # (4, 4)
        )
        conv_per_stage_encoder = (2, 2, 2, 2, 2, 2, 2)
        conv_per_stage_decoder = (2, 2, 2, 2, 2, 2)

        return PlainConvCBAMUNet.compute_approx_vram_consumption(patch_size, 32, 512, 4, 3, pool_op_kernel_sizes,
                                                             conv_per_stage_encoder, conv_per_stage_decoder, 2, 56)


if __name__ == "__main__":
    conv_op_kernel_sizes = ((3, 3),
                            (3, 3),
                            (3, 3),
                            (3, 3),
                            (3, 3),
                            (3, 3),
                            (3, 3))
    pool_op_kernel_sizes = ((1, 1),
                            (2, 2),
                            (2, 2),
                            (2, 2),
                            (2, 2),
                            (2, 2),
                            (2, 2))
    patch_size = (256, 256)
    batch_size = 56
    unet = PlainConvCBAMUNet(4, 32, (2, 2, 2, 2, 2, 2, 2), 2, pool_op_kernel_sizes, conv_op_kernel_sizes,
                         get_default_network_config(2, dropout_p=None), get_default_network_CBAM_config(2, dropout_p=None, nonlin="ReLU"),
                         4, (2, 2, 2, 2, 2, 2), False, False, max_features=512).cuda()
    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)

    unet.compute_reference_for_vram_consumption_3d()
    unet.compute_reference_for_vram_consumption_2d()

    dummy_input = torch.rand((batch_size, 4, *patch_size)).cuda()
    dummy_gt = (torch.rand((batch_size, 1, *patch_size)) * 4).round().clamp_(0, 3).cuda().long()

    optimizer.zero_grad()
    skips = unet.encoder(dummy_input)
    print([i.shape for i in skips])
    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
                    'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, {})
    output = unet.decoder(skips)

    l = loss(output, dummy_gt)
    l.backward()

    optimizer.step()

    import hiddenlayer as hl
    g = hl.build_graph(unet, dummy_input)
    g.save("/home/fabian/test.pdf")

    """conv_op_kernel_sizes = ((3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3),
                            (3, 3, 3))
    pool_op_kernel_sizes = ((1, 1, 1),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2),
                            (2, 2, 2))
    patch_size = (160, 128, 128)
    unet = PlainConvUNet(4, 32, (2, 2, 2, 2, 2, 2), 2, pool_op_kernel_sizes, conv_op_kernel_sizes,
                         get_default_network_config(3, dropout_p=None), 4, (2, 2, 2, 2, 2), False, False, max_features=512).cuda()
    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)

    unet.compute_reference_for_vram_consumption_3d()
    unet.compute_reference_for_vram_consumption_2d()

    dummy_input = torch.rand((2, 4, *patch_size)).cuda()
    dummy_gt = (torch.rand((2, 1, *patch_size)) * 4).round().clamp_(0, 3).cuda().long()

    optimizer.zero_grad()
    skips = unet.encoder(dummy_input)
    print([i.shape for i in skips])
    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'smooth_in_nom': True,
                    'do_bg': False, 'rebalance_weights': None, 'background_weight': 1}, {})
    output = unet.decoder(skips)

    l = loss(output, dummy_gt)
    l.backward()

    optimizer.step()

    import hiddenlayer as hl
    g = hl.build_graph(unet, dummy_input)
    g.save("/home/fabian/test.pdf")"""
