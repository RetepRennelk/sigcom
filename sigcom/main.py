import numpy as np
from utilities import generate_bits, qam_alphabet, \
    map_bits_to_symbol_alphabet, make_noise, \
    plot_constellation
from numba import njit

@njit
def _max_star(a, b):
    '''
    log(exp(a)+exp(b))
    '''
    return np.max(np.array([a, b])) + np.log(1+np.exp(-np.abs(a-b)))


@njit
def receiver(rx):
    Llrs = np.zeros(N_bits)
    for k in range(N_cells):
        D = np.zeros(M)
        for i in range(M):
            D[i] = -SNR*np.abs(rx[k]-qam[i])**2
        for m in range(ldM):
            num = den = np.float64(0)
            for i in range(M):
                if (i >> (ldM - 1 - m)) & 1:
                    den = _max_star(den, D[i])
                else:
                    num = _max_star(num, D[i])
            Llrs[k*ldM+m] = num - den
    return Llrs


SNR_dB = 20
N_cells = 10000
ldM = 8

N_bits = ldM * N_cells
M = 2**ldM

bits = generate_bits(N_bits)
qam = qam_alphabet(M)
tx = map_bits_to_symbol_alphabet(bits, qam)
noise = make_noise(int(N_bits/ldM))
SNR = 10**(SNR_dB/10)
rx = tx + noise / np.sqrt(SNR)
Llrs = receiver(rx)
plot_constellation(rx, 'bo')
