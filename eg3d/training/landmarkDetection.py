import numpy as np
import torch

from torch_utils import persistence
from training.networks_stylegan2 import DiscriminatorEpilogue, DiscriminatorBlock


@persistence.persistent_class
class LandmarkDetector(torch.nn.Module):
    def __init__(self,
                 nb_pts,
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=4,  # Use FP16 for the N highest resolutions.
                 conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 block_kwargs=None,  # Arguments for DiscriminatorBlock.
                 epilogue_kwargs=None,  # Arguments for DiscriminatorEpilogue.
                 ):
        super().__init__()
        if epilogue_kwargs is None:
            epilogue_kwargs = {}
        if block_kwargs is None:
            block_kwargs = {}
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=nb_pts, resolution=4, use_cmap=False, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, update_emas=False, **block_kwargs):
        _ = update_emas  # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        x = self.b4(x, img, None)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'
