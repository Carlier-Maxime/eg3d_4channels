import numpy as np
import torch
import torch.nn as nn

from torch_utils import persistence
from training.networks_stylegan2 import DiscriminatorEpilogue, DiscriminatorBlock


@persistence.persistent_class
class LandmarkDetector(nn.Module):
    def __init__(self,
                 nb_pts,
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 pts_dim: int = 3,
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
        self.nb_pts = nb_pts
        self.pts_dim = pts_dim
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
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=nb_pts * pts_dim, resolution=4, use_cmap=False, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, update_emas=False, **block_kwargs):
        _ = update_emas  # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        x = self.b4(x, img, None)
        return x.reshape(x.shape[0], self.nb_pts, self.pts_dim)

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'


@persistence.persistent_class
class LandmarkDetectorExperience(nn.Module):
    def __init__(self,
                 nb_pts,  # Number of points
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 pts_dim: int = 3,
                 **_
                 ):
        super().__init__()
        self.pts_dim = pts_dim
        embed_pooling = []
        res = img_resolution
        assert (res & (res - 1)) == 0 and res != 0, "img resolution is not a 2^x. Not supported"
        while res > 256:
            embed_pooling += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(img_channels, img_channels, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True)]
            res >>= 1
        self.embed = nn.Sequential(
            nn.Conv2d(img_channels, img_channels, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            *embed_pooling,
            nn.Conv2d(img_channels, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
        )

        to64res = []
        while res > 64:
            to64res += [nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)]
            res >>= 1
        self.to64res = nn.Sequential(*to64res)

        def channelsUp(input_channels: int, pool2D: bool = False) -> list[nn.Module]:
            sequence = [
                nn.Conv2d(input_channels, input_channels, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True)
            ]
            if pool2D:
                sequence += [nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)]
            return sequence

        to8res = []
        while res > 8:
            to8res += [nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)]
            res >>= 1

        self.features = nn.Sequential(
            *channelsUp(128, res > 32),
            *channelsUp(256, res > 16),
            *to8res,
            *channelsUp(512, res > 4),
            nn.Conv2d(1024, 1024, kernel_size=3, dilation=1, padding=1))
        res = min(res, 4)

        pow2_pts = 1 << (int(np.log2(nb_pts * pts_dim)) + 2)
        toPow2_pts = []
        channels = 1024
        while channels < pow2_pts:
            toPow2_pts.append(nn.Conv2d(channels, channels << 1, kernel_size=3, dilation=1, padding=1))
            toPow2_pts.append(nn.ReLU(inplace=True))
            channels <<= 1
            toPow2_pts.append(nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1))
            toPow2_pts.append(nn.ReLU(inplace=True))
        self.toPow2_pts = nn.Sequential(*toPow2_pts)

        self.fc_layers = nn.Sequential(
            nn.Linear(channels * res * res, pow2_pts), nn.ReLU(inplace=True),
            nn.Linear(pow2_pts, pow2_pts), nn.ReLU(inplace=True),
            nn.Linear(pow2_pts, nb_pts * pts_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.to64res(x)
        x = self.features(x)
        x = self.toPow2_pts(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_layers(x)
        return x.reshape(x.shape[0], x.shape[1] // self.pts_dim, -1)
