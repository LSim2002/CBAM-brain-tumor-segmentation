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


import numpy as np
import torch
from nnunet.network_architecture.custom_modules.conv_blocks import BasicResidualBlock
from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.generic_modular_UNet import PlainConvUNetDecoder, get_default_network_config
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
from torch.optim import SGD
from torch.backends import cudnn

#*****
from nnunet.network_architecture.custom_modules.cbam_conv_blocks import ResidualCBAMLayer

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




class ResidualUNetCBAMEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, cbam_props, default_return_skips=True,
                 max_num_features=480, block=BasicResidualBlock, block_kwargs=None):
        """
        Following UNet building blocks can be added by utilizing the properties this class exposes

        this one includes the bottleneck layer!

        :param input_channels:
        :param base_num_features:
        :param num_blocks_per_stage:
        :param feat_map_mul_on_downscale:
        :param pool_op_kernel_sizes:
        :param conv_kernel_sizes:
        :param props:
        """
        super(ResidualUNetCBAMEncoder, self).__init__()

        if block_kwargs is None:
            block_kwargs = {}

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

        self.initial_conv = props['conv_op'](input_channels, base_num_features, 3, padding=1, **props['conv_op_kwargs'])
        self.initial_norm = props['norm_op'](base_num_features, **props['norm_op_kwargs'])
        self.initial_nonlin = props['nonlin'](**props['nonlin_kwargs'])

        current_input_features = base_num_features
        for stage in range(num_stages):
            current_output_features = min(base_num_features * feat_map_mul_on_downscale ** stage, max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]

            current_stage = ResidualCBAMLayer(current_input_features, current_output_features, current_kernel_size, props, cbam_props,
                                          self.num_blocks_per_stage[stage], current_pool_kernel_size, block,
                                          block_kwargs)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_stage.output_channels)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_stage.output_channels

        self.output_features = current_input_features

        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, return_skips=None):
        """

        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        x = self.initial_nonlin(self.initial_norm(self.initial_conv(x)))
        for s in self.stages:
            x = s(x)
            if self.default_return_skips:
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        # print(x.shape)

        if return_skips:
            return skips
        else:
            return x

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        feat_map_mul_on_downscale, batch_size):
        npool = len(pool_op_kernel_sizes) - 1

        current_shape = np.array(patch_size)

        tmp = (num_conv_per_stage_encoder[0] * 2 + 1) * np.prod(current_shape) * base_num_features \
              + num_modalities * np.prod(current_shape)

        num_feat = base_num_features

        for p in range(1, npool + 1):
            current_shape = current_shape / np.array(pool_op_kernel_sizes[p])
            num_feat = min(num_feat * feat_map_mul_on_downscale, max_num_features)
            num_convs = num_conv_per_stage_encoder[p] * 2 + 1  # + 1 for conv in skip in first block
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat
        return tmp * batch_size





class FabiansCBAMUNet(SegmentationNetwork):
    """
    Residual Encoder, Plain conv decoder
    """
    use_this_for_2D_configuration = 1244233721.0  # 1167982592.0
    use_this_for_3D_configuration = 1230348801.0
    default_blocks_per_stage_encoder = (1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4)
    default_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    default_min_batch_size = 2 # this is what works with the numbers above

    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, cbam_props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, max_features=512, initializer=None,
                 block=BasicResidualBlock,
                 props_decoder=None, block_kwargs=None):
        super().__init__()

        if block_kwargs is None:
            block_kwargs = {}

        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = ResidualUNetCBAMEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                           feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes,
                                           props, cbam_props, default_return_skips=True, max_num_features=max_features,
                                           block=block, block_kwargs=block_kwargs)
        props['dropout_op_kwargs']['p'] = 0
        if props_decoder is None:
            props_decoder = props
        self.decoder = PlainConvUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props_decoder,
                                            deep_supervision, upscale_logits)
        if initializer is not None:
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    @staticmethod
    def compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, num_conv_per_stage_encoder,
                                        num_conv_per_stage_decoder, feat_map_mul_on_downscale, batch_size):
        enc = ResidualUNetCBAMEncoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                  num_modalities, pool_op_kernel_sizes,
                                                                  num_conv_per_stage_encoder,
                                                                  feat_map_mul_on_downscale, batch_size)
        dec = PlainConvUNetDecoder.compute_approx_vram_consumption(patch_size, base_num_features, max_num_features,
                                                                   num_classes, pool_op_kernel_sizes,
                                                                   num_conv_per_stage_decoder,
                                                                   feat_map_mul_on_downscale, batch_size)

        return enc + dec


def find_3d_configuration():
    # lets compute a reference for 3D
    # we select hyperparameters here so that we get approximately the same patch size as we would get with the
    # regular unet. This is just my choice. You can do whatever you want
    # These default hyperparemeters will then be used by the experiment planner

    # since this is more parameter intensive than the UNet, we will test a configuration that has a lot of parameters
    # herefore we copy the UNet configuration for Task005_Prostate
    cudnn.deterministic = False
    cudnn.benchmark = True

    patch_size = (20, 320, 256)
    max_num_features = 320
    num_modalities = 2
    num_classes = 3
    batch_size = 2

    # now we fiddle with the network specific hyperparameters until everything just barely fits into a titanx
    blocks_per_stage_encoder = FabiansCBAMUNet.default_blocks_per_stage_encoder
    blocks_per_stage_decoder = FabiansCBAMUNet.default_blocks_per_stage_decoder
    initial_num_features = 32

    # we neeed to add a [1, 1, 1] for the res unet because in this implementation all stages of the encoder can have a stride
    pool_op_kernel_sizes = [[1, 1, 1],
                            [1, 2, 2],
                            [1, 2, 2],
                            [2, 2, 2],
                            [2, 2, 2],
                            [1, 2, 2],
                            [1, 2, 2]]

    conv_op_kernel_sizes = [[1, 3, 3],
                            [1, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3],
                            [3, 3, 3]]

    unet = FabiansCBAMUNet(num_modalities, initial_num_features, blocks_per_stage_encoder[:len(conv_op_kernel_sizes)], 2,
                       pool_op_kernel_sizes, conv_op_kernel_sizes,
                       get_default_network_config(3, dropout_p=None), num_classes,
                       blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], False, False,
                       max_features=max_num_features).cuda()

    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)
    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    dummy_input = torch.rand((batch_size, num_modalities, *patch_size)).cuda()
    dummy_gt = (torch.rand((batch_size, 1, *patch_size)) * num_classes).round().clamp_(0, 2).cuda().long()

    for _ in range(20):
        optimizer.zero_grad()
        skips = unet.encoder(dummy_input)
        print([i.shape for i in skips])
        output = unet.decoder(skips)

        l = loss(output, dummy_gt)
        l.backward()

        optimizer.step()
        if _ == 0:
            torch.cuda.empty_cache()

    # that should do. Now take the network hyperparameters and insert them in FabiansUNet.compute_approx_vram_consumption
    # whatever number this spits out, save it to FabiansUNet.use_this_for_batch_size_computation_3D
    print(FabiansCBAMUNet.compute_approx_vram_consumption(patch_size, initial_num_features, max_num_features, num_modalities,
                                                num_classes, pool_op_kernel_sizes,
                                                blocks_per_stage_encoder[:len(conv_op_kernel_sizes)],
                                                blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], 2, batch_size))
    # the output is 1230348800.0 for me
    # I increment that number by 1 to allow this configuration be be chosen


def find_2d_configuration():
    # lets compute a reference for 3D
    # we select hyperparameters here so that we get approximately the same patch size as we would get with the
    # regular unet. This is just my choice. You can do whatever you want
    # These default hyperparemeters will then be used by the experiment planner

    # since this is more parameter intensive than the UNet, we will test a configuration that has a lot of parameters
    # herefore we copy the UNet configuration for Task003_Liver
    cudnn.deterministic = False
    cudnn.benchmark = True

    patch_size = (512, 512)
    max_num_features = 512
    num_modalities = 1
    num_classes = 3
    batch_size = 12

    # now we fiddle with the network specific hyperparameters until everything just barely fits into a titanx
    blocks_per_stage_encoder = FabiansCBAMUNet.default_blocks_per_stage_encoder
    blocks_per_stage_decoder = FabiansCBAMUNet.default_blocks_per_stage_decoder
    initial_num_features = 30

    # we neeed to add a [1, 1, 1] for the res unet because in this implementation all stages of the encoder can have a stride
    pool_op_kernel_sizes = [[1, 1],
                            [2, 2],
                            [2, 2],
                            [2, 2],
                            [2, 2],
                            [2, 2],
                            [2, 2],
                            [2, 2]]

    conv_op_kernel_sizes = [[3, 3],
                           [3, 3],
                           [3, 3],
                           [3, 3],
                           [3, 3],
                           [3, 3],
                           [3, 3],
                           [3, 3]]

    unet = FabiansCBAMUNet(num_modalities, initial_num_features, blocks_per_stage_encoder[:len(conv_op_kernel_sizes)], 2,
                       pool_op_kernel_sizes, conv_op_kernel_sizes,
                       get_default_network_config(2, dropout_p=None), num_classes,
                       blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], False, False,
                       max_features=max_num_features).cuda()

    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)
    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    dummy_input = torch.rand((batch_size, num_modalities, *patch_size)).cuda()
    dummy_gt = (torch.rand((batch_size, 1, *patch_size)) * num_classes).round().clamp_(0, 2).cuda().long()

    for _ in range(20):
        optimizer.zero_grad()
        skips = unet.encoder(dummy_input)
        print([i.shape for i in skips])
        output = unet.decoder(skips)

        l = loss(output, dummy_gt)
        l.backward()

        optimizer.step()
        if _ == 0:
            torch.cuda.empty_cache()

    # that should do. Now take the network hyperparameters and insert them in FabiansUNet.compute_approx_vram_consumption
    # whatever number this spits out, save it to FabiansUNet.use_this_for_batch_size_computation_2D
    print(FabiansCBAMUNet.compute_approx_vram_consumption(patch_size, initial_num_features, max_num_features, num_modalities,
                                                num_classes, pool_op_kernel_sizes,
                                                blocks_per_stage_encoder[:len(conv_op_kernel_sizes)],
                                                blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], 2, batch_size))
    # the output is 1244233728.0 for me
    # I increment that number by 1 to allow this configuration be be chosen
    # This will not fit with 32 filters, but so will the regular U-net. We still use 32 filters in training.
    # This does not matter because we are using mixed precision training now, so a rough memory approximation is OK


if __name__ == "__main__":
    pass

