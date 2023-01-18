from assets.lczero.lczero_weights_path import DEFAULT_LC_ZERO_WEIGHTS_PATH
from lczero_general_backend.lczero_classes import LCZeroBackend

from lczero_backend_athena.lczero_athena import Weights, Backend, GameState


lc_zero_local_backend = LCZeroBackend(Weights, Backend, GameState, DEFAULT_LC_ZERO_WEIGHTS_PATH)