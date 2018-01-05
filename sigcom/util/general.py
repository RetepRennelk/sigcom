import numpy as np


def int_to_bits(ints, bw):
    '''
    Convert integers 'ints' to bits with bitwidth 'bw'

    Parameters:
    - ints: list of integers
    - bw:  bitwidth to which the integers are converted
    '''
    l = [((i >> bw-1-j)) & 1 for i in ints for j in range(bw)]
    return np.array(l, dtype=np.int)


if __name__ == '__main__':
    ints = range(5)
    bits = int_to_bits(ints, 3)
    print(bits.reshape(-1, 3))
