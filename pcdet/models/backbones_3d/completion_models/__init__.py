import sys
from .model import PMPNet
sys.path.append('../pointnet2_ops_lib')
sys.path.append('..')

__all__ = {
    'PMPNet': PMPNet
}