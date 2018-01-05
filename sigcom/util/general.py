import numpy as np


def int_to_bits(ints, m):
    '''
    Parameters:
    - ints: list of integers
    - m:  bitwidth to which the integers are converted
    '''
    l = [((i >> m-1-j)) & 1 for i in ints for j in range(m)]
    return np.array(l, dtype=np.int)


if __name__ == '__main__':
    ints = range(5)
    bits = int_to_bits(ints, 3)
    print(bits.reshape(-1, 3))
