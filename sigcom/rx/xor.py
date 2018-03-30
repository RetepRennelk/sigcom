import numpy as np
from numba import njit


@njit
def _clip(inp):
    inp[inp > 0] = 1
    inp[inp < 0] = -1


@njit
def XorFwdBwd(Llrs):
    N = len(Llrs)

    bwd = np.ones(N)
    for k in range(N-1, 0, -1):
        bwd[k-1] = bwd[k] * Llrs[k]
    _clip(bwd)

    fwd = np.ones(N)
    for k in range(N-1):
        fwd[k+1] = fwd[k] * Llrs[k]
    _clip(fwd)

    return fwd * bwd
