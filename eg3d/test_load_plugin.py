from torch_utils import custom_ops
custom_ops.verbosity = 'full'
from torch_utils.ops import bias_act, upfirdn2d
bias_act._init()
upfirdn2d._init()
