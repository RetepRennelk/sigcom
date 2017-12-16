import numpy as np
from numba import njit


@njit
def _breadth_first_search(mtx, n):
    M = mtx.shape[0]
    l0 = mtx[:, n]
    while 1:
        l1 = np.copy(l0)
        check_nodes = np.nonzero(l1)[0]
        for c in check_nodes:
            bit_nodes = np.nonzero(mtx[c, :])[0]
            l1 += mtx[:, bit_nodes].sum(axis=1)
        l1 = np.array([np.int32(x != 0) for x in l1])
        N0 = np.sum(l0)
        N1 = np.sum(l1)
        if (N0 < M and N1 == M) or (N0 >= N1 and N0 < M):
            break
        l0 = l1
    return l0


def _find_cn(mtx, neighborhood, sw_rand):
    zero_idx = np.where(neighborhood == 0)[0]
    h_sum = mtx[zero_idx, :].sum(axis=1)
    if sw_rand:
        mins = np.nonzero(h_sum == np.min(h_sum))[0]
        idx = np.random.choice(mins)
    else:
        idx = np.argmin(h_sum)
    return zero_idx[idx]


def peg_v0(M, N, dv, sw_rand):
    '''
    Progressive Edge Growth - Version 0

    This is a first version with constant column
    weight dv and either fixed selection of the next
    check node (sw_rand=0) or random selection of the
    next check node (sw_rand=1).
    '''
    mtx = np.zeros((M, N), dtype=np.int32)
    for n in range(N):
        v_sum = mtx.sum(axis=1)
        m = np.argmin(v_sum)
        mtx[m, n] = 1
        for d in range(1, dv):
            nh = _breadth_first_search(mtx, n)
            idx = _find_cn(mtx, nh, sw_rand)
            mtx[idx, n] = 1
    return mtx


if __name__ == '__main__':
    M = 10
    N = 20
    dv = 2
    mtx = peg_v0(M, N, dv, sw_rand=0)
    import matplotlib.pyplot as plt
    plt.spy(mtx)
    plt.show()
