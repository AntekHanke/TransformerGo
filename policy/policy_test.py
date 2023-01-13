import os

from policy.cllp import CLLP


def test_cllp():
    if "LOCAL_CLLP_PATH_FOR_TESTING" in os.environ:
        cllp = CLLP(os.environ[""])
