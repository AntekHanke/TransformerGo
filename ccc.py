from mpmath import mp, mpf, fmod
import hashlib

mp.dps = 50

def hash_string_to_int(arg):
    return int(hashlib.sha256(arg.encode("utf-8")).hexdigest(), 16) % 10**30

def hash_string_to_float(arg):
    assert mp.dps >= 50
    x = mpf(hash_string_to_int(arg))
    return fmod(x * mp.pi, 1)

def get_split(arg):
    float_hash = hash_string_to_float(arg)
    if float_hash < 0.95:
        return "train"
    elif float_hash < 0.96:
        return "val"
    else:
        return "test"


for i in range(100):
    print(f"{i} -> {hash_string_to_float(str(i))}")
