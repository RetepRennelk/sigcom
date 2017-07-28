import numpy as np

def map_bits_to_int(bits, ldM):
    '''
    [Syntax]
    map_bits_to_int(bits, ldM)
    '''
    N_bits = len(bits)
    assert N_bits%ldM==0
    N = int(N_bits / ldM)
    ids = np.zeros(N)
    for k in np.arange(N):
        id = 0
        for m in np.arange(ldM):
            id += bits[k*ldM+m]*2**(ldM-1-m)
        ids[k] = id
    return ids    
    
if __name__ == "__main__":
    pass
