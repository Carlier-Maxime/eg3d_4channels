import torch.nn
from .triplane import TriPlaneGenerator
from .landmarkDetection import LandmarkDetectorExperience


class Generator(torch.nn.Module):
    def __init__(self,
                 z_dim,  # Input latent (Z) dimensionality.
                 c_dim,  # Conditioning label (C) dimensionality.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output resolution.
                 img_channels,  # Number of output color channels.
                 lmks_dim: int = 0,  # Landmarks dimensionality (Number of landmarks)
                 sr_num_fp16_res=0,
                 mapping_kwargs=None,  # Arguments for MappingNetwork.
                 rendering_kwargs=None,
                 sr_kwargs=None,
                 **synthesis_kwargs,  # Arguments for SynthesisNetwork.
                 ):
        super().__init__()
        self.triplane = TriPlaneGenerator(z_dim, c_dim, w_dim, img_resolution, img_channels, sr_num_fp16_res, mapping_kwargs, rendering_kwargs, sr_kwargs, **synthesis_kwargs)
        self.lmkDetector = None
        if lmks_dim > 0:
            self.lmkDetector = LandmarkDetectorExperience(lmks_dim, **synthesis_kwargs)

    def forward(self, z: torch.Tensor, c: torch.Tensor, **triplaneGeneratorKwargs):
        if self.lmkDetector is None:
            d = self.triplane(z, c, **triplaneGeneratorKwargs)
            d["lmks"] = None
            return d
        ws = self.mapping(z, c, **triplaneGeneratorKwargs)
        planes = self.triplane.backbone.synthesis(ws, **triplaneGeneratorKwargs)
        lmks = self.lmkDetector(planes)
        d = self.triplane.synthesis(ws, planes=planes, **triplaneGeneratorKwargs)
        d["lmks"] = d
        return d

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            return getattr(self.triplane, item)
