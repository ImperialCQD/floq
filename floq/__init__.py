from . import core, optimization, systems, parallel
from .systems import System
from .errors import *

System.__module__ = __name__
