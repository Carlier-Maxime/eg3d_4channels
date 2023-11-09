import numpy as np
import torch
from torch_utils import persistence
from torch_utils.ops import bias_act


@persistence.persistent_class
class FullyConnectedLayerBase(torch.nn.Module):
    def __init__(self,
                 in_features,  # Number of input features.
                 out_features,  # Number of output features.
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier: float = 1.,  # Learning rate multiplier.
                 ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        self.weight = None
        self.bias = None

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias

        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b *= self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'