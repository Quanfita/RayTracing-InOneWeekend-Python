import math
import numba

@numba.jit
def unit_vector(v):
    return v / math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

@numba.jit
def squared_length(v):
    return v[0] ** 2 + v[1] ** 2 + v[2] ** 2

@numba.jit
def length(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)