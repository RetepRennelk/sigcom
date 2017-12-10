import numpy as np
from numba import njit


def encode(bits, cp, layerwise_pcks, bil):
    @njit
    def parity(bits, codebits, N, K, pck, nCyclicFactor, l):
        N_bits = len(bits)
        N_codewords = int(N_bits/K)
        N_layers = int((N-K)/nCyclicFactor)
        base_addr = np.arange(K)
        for n in np.arange(N_codewords):
            codebits[n*N + base_addr] = bits[n*K + base_addr]
            for m in np.arange(nCyclicFactor):
                addr = pck[:-2, 0] + (pck[:-2, 1] + m) % nCyclicFactor
                codebits[n*N+K+m*N_layers+l] = bits[n*K+addr].sum() % 2
        return codebits

    @njit
    def acc(codebits, N, K):
        N_codebits = len(codebits)
        N_codewords = int(N_codebits/N)
        addr = np.arange(N-K)
        for k in np.arange(N_codewords):
            parity = np.cumsum(codebits[k*N+K+addr]) % 2
            codebits[k*N+K+addr] = parity

    @njit
    def apply_bil(codebits, bil):
        N_codebits = len(codebits)
        N_ldpc = len(bil)
        N_codewords = int(N_codebits/N_ldpc)
        addr = np.arange(N_ldpc)
        for k in np.arange(N_codewords):
            codebits[k*N_ldpc+addr] = codebits[k*N_ldpc+bil]

    N_bits = len(bits)
    N_codebits = int(N_bits/cp.CR)
    codebits = np.zeros(N_codebits)
    for i, pck in enumerate(layerwise_pcks):
        codebits = parity(bits, codebits, cp.N, cp.K, pck, cp.nCyclicFactor, i)
    acc(codebits, cp.N, cp.K)
    apply_bil(codebits, bil)

    return codebits % 2


if __name__ == '__main__':
    #from sigcom.coding.atsc.code_param_short import get
    from sigcom.coding.test.code_param import get
    from sigcom.coding.PCM import PCM
    from sigcom.coding.util import get_layerwise_pck
    from sigcom.coding.util import get_parity_interleaver
    from sigcom.tx.util import generate_bits
    import matplotlib.pyplot as plt
    np.random.seed(0)
    cp = get([1, 2])
    #cp = get([8,15])
    pcm = PCM(cp)
    N_codewords = 2
    N_bits = N_codewords*cp.K
    bits = generate_bits(N_bits)
    bil = get_parity_interleaver(cp.K, cp.N, cp.nCyclicFactor)

    isParityPermuted = True
    H = pcm.make_layered(isParityPermuted)

    layerwise_pcks, diagOffsets = get_layerwise_pck(cp, isParityPermuted)

    codebits = encode(bits, cp, layerwise_pcks, bil)

    print((H.dot(codebits[:cp.N]) % 2).sum())
    print((H.dot(codebits[cp.N:]) % 2).sum())


