from lczero.lczero_weights_path import DEFAULT_LC_ZERO_WEIGHTS_PATH
from lczero_backend_athena.backends import Weights, Backend, GameState

from lczero_general_backend.lczero_classes import LCZeroBackend

lc_zero_athena_backend = LCZeroBackend(Weights, Backend, GameState, DEFAULT_LC_ZERO_WEIGHTS_PATH)