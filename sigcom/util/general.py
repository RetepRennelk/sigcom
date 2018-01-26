import numpy as np
import functools


def ints_to_bits(ints, bw):
    '''
    Convert integers 'ints' to bits with bitwidth 'bw'

    Parameters:
    - ints: list of integers
    - bw:  bitwidth to which the integers are converted
    '''
    l = [((i >> bw-1-j)) & 1 for i in ints for j in range(bw)]
    return np.array(l, dtype=np.int)


def bits_to_ints(bits, bw):
    '''
    Convert bits to integers with bitwidth 'bw'
    '''
    weights = 2**np.arange(bw)[::-1]
    ints = bits.reshape(-1, bw).dot(weights)
    return np.array(ints, np.int)


@functools.lru_cache(maxsize=None)
def hermite_poly(n):
    '''
    H0(x) = 1
    H1(x) = 2x
    H_(n)(x) = 2*x*H_(n-1)(x) - 2*(n-1)*H_(n-2)(x)
    '''
    if n == 0:
        return np.array([1.])
    elif n == 1:
        return np.array([2., 0.])
    else:
        out = np.append(hermite_poly(n-1), 0.)
        out[2:] -= (n-1)*hermite_poly(n-2)
        return 2*out


def gauss_hermite_weights_abscissas(n=32):
    '''
    wi, xi = gauss_hermite_weights_abscissas(n=32)
    '''
    H_n = hermite_poly(n)
    xi = np.roots(H_n)

    H_n_1 = hermite_poly(n-1)
    num = 2**(n-1) * np.math.factorial(n-1) * np.sqrt(np.pi)
    den = n * np.polyval(H_n_1, xi)**2
    wi = num / den
    return wi, xi


if __name__ == '__main__':
    ints = range(5)
    bits = ints_to_bits(ints, 3)
    print(bits.reshape(-1, 3))
