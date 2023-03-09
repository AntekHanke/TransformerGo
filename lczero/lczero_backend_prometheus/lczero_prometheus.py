from lczero.lczero_weights_path import DEFAULT_LC_ZERO_WEIGHTS_PATH
from lczero.lczero_backend_prometheus.backends import Weights, Backend, GameState

from lczero.lczero_general_backend.lczero_classes import LCZeroBackend

lc_zero_prometheus_backend = LCZeroBackend(Weights, Backend, GameState, DEFAULT_LC_ZERO_WEIGHTS_PATH)
