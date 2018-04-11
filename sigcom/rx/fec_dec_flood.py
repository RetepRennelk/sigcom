import numpy as np
from numba import njit
from sigcom.rx.softxor import softXorFwdBwd


@njit
def _CN_proc(N, K, indptr, data):
    # CN Processing
    for r in range(N - K):
        fr = indptr[r]
        to = indptr[r+1]
        data[fr:to] = softXorFwdBwd(data[fr:to])


def fec_dec_flood(Llrs, ldpcEncAtsc, max_iterations):
    '''
    FEC decoding with the SPA for LDPC codewords using flooding
    '''
    K = ldpcEncAtsc.cp.K
    N_ldpc = ldpcEncAtsc.cp.N
    N_codewords = int(len(Llrs) / N_ldpc)
    detected = np.zeros((N_codewords, N_ldpc))
    m_Llrs = Llrs.reshape(N_codewords, -1)

    for k in range(N_codewords):
        messages = ldpcEncAtsc.H_dec.copy()
        messages.data.fill(0)

        for it in range(max_iterations+1):
            # VN Processing
            L_sum = m_Llrs[k] + messages.sum(axis=0).A1

            syndrom = ldpcEncAtsc.H_dec.dot(L_sum < 0)
            N_err = np.sum(np.asarray(syndrom, np.int) % 2)
            if N_err == 0:
                break

            messages.data = L_sum[messages.indices] - messages.data

            _CN_proc(N_ldpc, K, messages.indptr, messages.data)

        detected[k] = L_sum
    return detected.reshape(-1)
