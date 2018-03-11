import numpy as np
from numba import njit
from sigcom.rx.xor import XorFwdBwd


@njit
def _CN_proc(N, K, indptr, data):
    # CN Processing
    for r in range(N - K):
        fr = indptr[r]
        to = indptr[r+1]
        data[fr:to] = XorFwdBwd(data[fr:to])


def fec_dec_bec(rx, ldpcEncAtsc, max_iterations):
    '''
    FEC decoding with the SPA for LDPC codewords
    received over a BEC channel.
    '''
    K = ldpcEncAtsc.cp.K
    N_ldpc = ldpcEncAtsc.cp.N
    N_codewords = int(len(rx) / N_ldpc)
    detected = np.zeros((N_codewords, N_ldpc))
    m_rx = rx.reshape(N_codewords, -1)

    for k in range(N_codewords):
        messages = ldpcEncAtsc.H_dec.copy()
        messages.data.fill(0)

        for it in range(max_iterations):
            # VN Processing
            L_sum = m_rx[k] + messages.sum(axis=0).A1

            syndrom = ldpcEncAtsc.H_dec.dot(L_sum < 0)
            N_err = np.sum(np.asarray(syndrom, np.int) % 2)
            if N_err == 0:
                break

            messages.data = L_sum[messages.indices] - messages.data

            _CN_proc(N_ldpc, K, messages.indptr, messages.data)

        detected[k] = L_sum
    return detected
