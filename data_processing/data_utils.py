from mpmath import mp, mpf, fmod
import hashlib

mp.dps = 50
TRAIN_TEST_SPLIT_SEED = 11

def hash_string_to_int(arg):
    arg = str(arg) + str(TRAIN_TEST_SPLIT_SEED)
    return int(hashlib.sha256(arg.encode("utf-8")).hexdigest(), 16) % 10**30

def hash_string_to_float(arg):
    assert mp.dps >= 50
    x = mpf(hash_string_to_int(arg))
    return fmod(x * mp.pi, 1)

def get_split(arg, train_eval_split):
    float_hash = hash_string_to_float(arg)
    if float_hash < train_eval_split:
        return "train"
    else:
        return "eval"