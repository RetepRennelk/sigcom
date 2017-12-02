import numpy as np
from numba import njit


def make_noise(N):
    return 1/np.sqrt(2)*(np.random.randn(N) + 1j * np.random.randn(N))


@njit
def _max_star(a, b):
    '''
    log(exp(a)+exp(b))
    '''
    return np.max(np.array([a, b])) + np.log(1+np.exp(-np.abs(a-b)))


@njit
def demap(rx, qam, SNR, La=np.array([])):
    N_cells = len(rx)
    M = len(qam)
    ldM = int(np.log2(M))
    N_bits = N_cells*ldM
    Llrs = np.zeros(N_bits)
    for k in range(N_cells):
        D = np.zeros(M)
        for i in range(M):
            D[i] = -SNR*np.abs(rx[k]-qam[i])**2
            if len(La) > 0:
                for m in range(ldM):
                    bit = (i >> (ldM - 1 - m)) & 1
                    D[i] -= bit*La[k*ldM+m]
        for m in range(ldM):
            num = den = np.float64(-np.inf)
            for i in range(M):
                if (i >> (ldM - 1 - m)) & 1:
                    den = _max_star(den, D[i])
                else:
                    num = _max_star(num, D[i])
            Llrs[k*ldM+m] = num - den
    return Llrs


if __name__ == "__main__":
    pass
