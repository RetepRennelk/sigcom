import numpy as np


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


if __name__ == '__main__':
    ints = range(5)
    bits = ints_to_bits(ints, 3)
    print(bits.reshape(-1, 3))
