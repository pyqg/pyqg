__version__='0.1.3'
from .model import Model
from .qg_model import QGModel
from .bt_model import BTModel
from .sqg_model import SQGModel
from .layered_model import LayeredModel
from .particles import LagrangianParticleArray2D, GriddedLagrangianParticleArray2D

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
