from .USNID import USNIDModel
from .MCN import MCNModel
from .CC import CCModel
from .SCCL import SCCLModel
from .UMC_ICLA import UMCICLAModel

methods_map = {
    'usnid': USNIDModel,
    'mcn': MCNModel,
    'cc': CCModel,
    'sccl': SCCLModel,
    'umc-icla': UMCICLAModel,
}
