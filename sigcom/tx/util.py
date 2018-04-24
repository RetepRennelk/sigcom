import numpy as np
import matplotlib.pyplot as plt
from numba import njit


@njit
def generate_bits(N_bits):
    '''
    [Syntax]
    bits = generate_bits(N_bits)

    [Remarks]
    Sugar coating for the call 'np.random.randint(0, 2, N_bits)'
    '''
    return np.random.randint(0, 2, N_bits)


def _map_bits_to_int(bits, ldM):
    '''
    [Syntax]
    _map_bits_to_int(bits, ldM)
    '''
    weights = 2**np.arange(ldM)[::-1]
    return bits.reshape(-1, ldM).dot(weights)


def map_bits_to_symbol_alphabet(bits, alphabet):
    '''
    [Syntax]
    tx = map_bits_to_symbol_alphabet(bits, alphabet):
    '''
    M = len(alphabet)
    tx = alphabet[_map_bits_to_int(bits, int(np.log2(M)))]
    return tx


def pam_alphabet(M):
    '''
    returns a gray-coded pam alphabet with M spots and unit variance
    '''
    linear_scale = np.arange(0, M)
    spots = np.flip(linear_scale, axis=0) - np.mean(linear_scale)
    spots = spots/np.std(spots)
    gray_code = (linear_scale >> 1) ^ linear_scale
    spots_gray = np.zeros(M)
    spots_gray[gray_code] = spots
    return spots_gray


def _split_integer(integer, bit_width):
    left_mask = 0
    right_mask = 0
    for i in range(bit_width >> 1):
        right_mask += (integer & 1) << i
        integer >>= 1
        left_mask += (integer & 1) << i
        integer >>= 1
    return left_mask, right_mask


def qam_alphabet(M):
    indices_re = np.zeros(M, dtype=np.int)
    indices_im = np.zeros(M, dtype=np.int)
    for i in range(M):
        indices_re[i], indices_im[i] = _split_integer(i, int(np.log2(M)))
    pam = pam_alphabet(int(np.sqrt(M)))
    qam = 1/np.sqrt(2)*np.array(pam[indices_re] + 1j * pam[indices_im])
    return qam


def plot_constellation(c, marker='bo'):
    plt.plot(c.real, c.imag, marker)
    ldM = int(np.log2(len(c)))
    for i in range(len(c)):
        str = ''.join(['1' if i & (1 << m) else '0' for m in range(ldM-1, -1, -1)])
        plt.text(c[i].real, c[i].imag, str)
    plt.axis('square')
    plt.grid()
    plt.show()


def make_cells(qam, N_cells):
    '''
    tx, bits = make_cells(qam, N_cells)
    '''
    M = len(qam)
    ldM = int(np.log2(M))
    N_bits = ldM*N_cells
    bits = generate_bits(N_bits)
    tx = map_bits_to_symbol_alphabet(bits, qam)
    return tx, bits


if __name__ == "__main__":
    z = generate_bits(10)
    pass
