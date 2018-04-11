from numba import njit
import numpy as np
from sigcom.rx.util import _max_star


@njit
def MI(rx, tx, X, h, P_noise):
    N_cells = len(rx)
    M0 = len(X)
    MI = 0
    for k in range(N_cells):
        D = rx[k] - tx[k]*h[k]
        num = -1/P_noise*np.abs(D)**2
        den = -np.inf
        for x in X:
            D = rx[k] - x*h[k]
            den = _max_star(den, -1/P_noise*np.abs(D)**2)
        MI += num - den
    return np.log2(M0)+MI/N_cells/np.log(2.0)
