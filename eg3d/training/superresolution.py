# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Superresolution network architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import torch
from torch_utils.ops import upfirdn2d
from torch_utils import persistence

from training.networks_stylegan2 import SynthesisBlock


@persistence.persistent_class
class SynthesisBlockNoUp(SynthesisBlock):
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 out_channels,  # Number of output channels.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of output color channels.
                 is_last,  # Is this the last block?
                 architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
                 resample_filter=None,  # Low-pass filter to apply when resampling activations.
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 use_fp16=False,  # Use FP16 for this block?
                 fp16_channels_last=False,  # Use channels-last memory format with FP16?
                 fused_modconv_default=True,  # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
                 **layer_kwargs,  # Arguments for SynthesisLayer.
                 ):
        super().__init__(in_channels, out_channels, w_dim, resolution, img_channels, is_last,
                         architecture=architecture,
                         resample_filter=resample_filter,
                         conv_clamp=conv_clamp,
                         use_fp16=use_fp16,
                         fp16_channels_last=fp16_channels_last,
                         fused_modconv_default=fused_modconv_default,
                         is_up=False,
                         **layer_kwargs)


# ----------------------------------------------------------------------------

class SuperresolutionHybridBase(torch.nn.Module):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias, img_channels,
                 block0_no_up, block0_out_channels, block0_resolution,
                 block1_out_channels, block1_resolution,
                 input_resolution=128, interpolate_condition="not_equal", num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,  # IGNORE
                 **block_kwargs):
        super().__init__()
        assert img_resolution == block1_resolution
        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = input_resolution
        self.sr_antialias = sr_antialias
        self.interpolate_condition = interpolate_condition
        self.block0 = SynthesisBlockNoUp if block0_no_up else SynthesisBlock
        self.block0 = self.block0(channels, block0_out_channels, w_dim=512, resolution=block0_resolution,
                                  img_channels=img_channels, is_last=False, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.block1 = SynthesisBlock(block0_out_channels, block1_out_channels, w_dim=512, resolution=block1_resolution,
                                     img_channels=img_channels, is_last=True, use_fp16=use_fp16, conv_clamp=(256 if use_fp16 else None), **block_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1, 3, 3, 1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        condition = False
        if self.interpolate_condition == "not_equal":
            condition = x.shape[-1] != self.input_resolution
        elif self.interpolate_condition == "less_than":
            condition = x.shape[-1] < self.input_resolution
        elif self.interpolate_condition == "great_than":
            condition = x.shape[-1] > self.input_resolution
        elif self.interpolate_condition == "equal":
            condition = x.shape[-1] > self.input_resolution
        if condition:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb


# for 512x512 generation
@persistence.persistent_class
class SuperresolutionHybrid8X(SuperresolutionHybridBase):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias, img_channels,
                 num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,  # IGNORE
                 **block_kwargs):
        super().__init__(channels, img_resolution, sr_num_fp16_res, sr_antialias, img_channels=img_channels,
                         block0_no_up=False, block0_out_channels=128, block0_resolution=256, block1_out_channels=64, block1_resolution=512,
                         num_fp16_res=num_fp16_res, conv_clamp=conv_clamp, channel_base=channel_base, channel_max=channel_max,  # IGNORE
                         **block_kwargs)


# ----------------------------------------------------------------------------

# for 256x256 generation
@persistence.persistent_class
class SuperresolutionHybrid4X(SuperresolutionHybridBase):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias, img_channels,
                 num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,  # IGNORE
                 **block_kwargs):
        super().__init__(channels, img_resolution, sr_num_fp16_res, sr_antialias, img_channels=img_channels,
                         block0_no_up=True, block0_out_channels=128, block0_resolution=128, block1_out_channels=64, block1_resolution=256,
                         interpolate_condition="less_than", num_fp16_res=num_fp16_res, conv_clamp=conv_clamp, channel_base=channel_base, channel_max=channel_max,  # IGNORE
                         **block_kwargs)


# ----------------------------------------------------------------------------

# for 128 x 128 generation
@persistence.persistent_class
class SuperresolutionHybrid2X(SuperresolutionHybridBase):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias, img_channels,
                 num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,  # IGNORE
                 **block_kwargs):
        super().__init__(channels, img_resolution, sr_num_fp16_res, sr_antialias, img_channels=img_channels,
                         block0_no_up=True, block0_out_channels=128, block0_resolution=64, block1_out_channels=64, block1_resolution=128,
                         input_resolution=64, num_fp16_res=num_fp16_res, conv_clamp=conv_clamp, channel_base=channel_base, channel_max=channel_max,  # IGNORE
                         **block_kwargs)


# ----------------------------------------------------------------------------

# TODO: Delete (here for backwards compatibility with old 256x256 models)
@persistence.persistent_class
class SuperresolutionHybridDeepfp32(SuperresolutionHybridBase):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, img_channels,
                 num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,  # IGNORE
                 **block_kwargs):
        super().__init__(channels, img_resolution, sr_num_fp16_res, False, img_channels=img_channels,
                         block0_no_up=True, block0_out_channels=128, block0_resolution=128, block1_out_channels=64, block1_resolution=256,
                         interpolate_condition="less_than", num_fp16_res=num_fp16_res, conv_clamp=conv_clamp, channel_base=channel_base, channel_max=channel_max,  # IGNORE
                         **block_kwargs)


# ----------------------------------------------------------------------------

# for 512x512 generation
@persistence.persistent_class
class SuperresolutionHybrid8XDC(SuperresolutionHybridBase):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias, img_channels,
                 num_fp16_res=4, conv_clamp=None, channel_base=None, channel_max=None,  # IGNORE
                 **block_kwargs):
        super().__init__(channels, img_resolution, sr_num_fp16_res, sr_antialias, img_channels=img_channels,
                         block0_no_up=False, block0_out_channels=256, block0_resolution=256, block1_out_channels=128, block1_resolution=512,
                         num_fp16_res=num_fp16_res, conv_clamp=conv_clamp, channel_base=channel_base, channel_max=channel_max,  # IGNORE
                         **block_kwargs)

# ----------------------------------------------------------------------------
