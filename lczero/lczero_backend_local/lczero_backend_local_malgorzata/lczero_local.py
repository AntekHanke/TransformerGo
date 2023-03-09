from lczero.lczero_weights_path import DEFAULT_LC_ZERO_WEIGHTS_PATH
from lczero.lczero_general_backend.lczero_classes import LCZeroBackend

from lczero.lczero_backend_local.lczero_backend_local_malgorzata.backends import Weights, Backend, GameState


lc_zero_local_backend = LCZeroBackend(Weights, Backend, GameState, DEFAULT_LC_ZERO_WEIGHTS_PATH)
