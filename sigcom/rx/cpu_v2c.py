import numpy as np


def update_v2c(Llrs, C2Vs, code, N_codewords):
    '''
    Llrs_ext = update_v2c(Llrs, C2Vs, code, N_codewords)

    Use this function as slow but correct variant updating
    Variable to Check node messages
    '''

    Llrs_ext = Llrs.copy()
    N_layers = len(code.layerwise_pcks)
    layers = np.append(np.arange(1, N_layers), 0)
    for k in range(N_codewords):
        N_offset = len(code.layerwise_pcks[0])
        for l in layers:
            pcks = code.layerwise_pcks[l]
            for i, pck in enumerate(pcks):
                for tx in range(code.nCyclicFactor):
                    a = pck[0] + (pck[1] + tx) % code.nCyclicFactor
                    Llrs_ext[k*code.N+a] += C2Vs[k*code.N_diags+N_offset+i, tx]
            N_offset = (N_offset + len(pcks)) % code.N_diags
    return Llrs_ext


if __name__ == '__main__':
    from sigcom.coding.atsc import code_param_long

    CR = [8, 15]
    N_codewords = 2
    code = code_param_long.get(CR)

    Llrs = np.random.randn(code.N*N_codewords)
    Llrs = np.asarray(Llrs, dtype=np.float32)
    C2Vs = np.random.randn(code.N_diags*N_codewords, code.nCyclicFactor)
    C2Vs = np.asarray(C2Vs, dtype=np.float32)

    Llrs_ext = update_v2c(Llrs, C2Vs, code, N_codewords)
    print(np.sum((Llrs_ext-Llrs)**2))
