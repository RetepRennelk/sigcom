import numpy as np
from sigcom.tx.util import generate_bits, qam_alphabet, \
    map_bits_to_symbol_alphabet, plot_constellation
from sigcom.rx.util import make_noise, demap


if __name__ == "__main__":
    SNR_dB = 20
    N_cells = 10000
    ldM = 6

    N_bits = ldM * N_cells
    M = 2**ldM

    bits = generate_bits(N_bits)
    qam = qam_alphabet(M)
    tx = map_bits_to_symbol_alphabet(bits, qam)
    noise = make_noise(int(N_bits/ldM))
    SNR = 10**(SNR_dB/10)
    rx = tx + noise / np.sqrt(SNR)
    Llrs = demap(rx, qam, SNR   )

    plot_constellation(rx, 'bo')
