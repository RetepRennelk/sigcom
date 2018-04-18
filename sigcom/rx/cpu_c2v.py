import numpy as np
from sigcom.rx.softxor import _partialSoftXor


def update_c2v(Llrs_ext, C2Vs, code, N_codewords):
    N_layers = len(code.layerwise_pcks)
    layers = np.append(np.arange(1, N_layers), 0)
    for k in range(N_codewords):
        N_offset = len(code.layerwise_pcks[0])
        for l in layers:
            pcks = code.layerwise_pcks[l]
            N = len(pcks)

            fwd = np.zeros((code.nCyclicFactor, len(pcks)))
            fwd[:, 0] = np.inf
            for i in range(N-1):
                for q in range(code.nCyclicFactor):
                    a = k*code.N + pcks[i, 0] + (pcks[i, 1]+q) % code.nCyclicFactor
                    fwd[q, i+1] = _partialSoftXor(fwd[q, i],
                                                  Llrs_ext[a]-C2Vs[k*code.N_diags+N_offset+i, q])

            bwd = np.zeros((code.nCyclicFactor, len(pcks)))
            bwd[:, -1] = np.inf
            for i in range(N-1, 0, -1):
                for q in range(code.nCyclicFactor):
                    a = k*code.N + pcks[i, 0] + (pcks[i, 1]+q) % code.nCyclicFactor
                    bwd[q, i-1] = _partialSoftXor(bwd[q, i],
                                                  Llrs_ext[a]-C2Vs[k*code.N_diags+N_offset+i, q])

            for i in range(N):
                for q in range(code.nCyclicFactor):
                    C2Vs[k*code.N_diags+N_offset+i, q] = _partialSoftXor(fwd[q, i], bwd[q, i])

            N_offset = (N_offset + len(pcks)) % code.N_diags


if __name__ == '__main__':
    from sigcom.coding.atsc import code_param_long

    CR = [8, 15]
    N_codewords = 2
    code = code_param_long.get(CR)

    Llrs_ext = np.random.randn(code.N*N_codewords)
    Llrs_ext = np.asarray(Llrs_ext, dtype=np.float32)
    C2Vs = np.random.randn(code.N_diags*N_codewords, code.nCyclicFactor)
    C2Vs = np.asarray(C2Vs, dtype=np.float32)
    C2Vs_old = C2Vs.copy()

    Llrs_ext = update_c2v(Llrs_ext, C2Vs, code, N_codewords)
    print(np.sum((C2Vs_old-C2Vs)**2))
    print(np.sum((C2Vs_old[:,0]-C2Vs[:,0])**2))
