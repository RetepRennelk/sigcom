import math
from numba import cuda
import numpy as np
from sigcom.cuda.util import _partialSoftXor


class C2V():
    def __init__(self, code, N_codewords):
        self.code = code
        self.N_codewords = N_codewords
        
        s = '''
ar_layerwise_pcks = np.array({ar_layerwise_pcks})
ar_addr = np.array({ar_addr})

@cuda.jit
def d_update_c2v(dLlrs_ext, dC2Vs):
    pcks = cuda.const.array_like(ar_layerwise_pcks)
    addr = cuda.const.array_like(ar_addr)
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    fwd = cuda.local.array({N_max_num_diags}, dtype=numba.float32)
    bwd = cuda.local.array({N_max_num_diags}, dtype=numba.float32)       

    fwd[0] = np.inf
    
    N_layers = len(addr)-1
    N_offset = 0
    for l in range(N_layers-1):
        N_pcks = (addr[l+1] - addr[l]) // 2
        
        # Forward Iteration
        for i in range(N_pcks-1):
            base   = pcks[N_offset+2*i]
            offset = pcks[N_offset+2*i+1]
            a      = base + (offset+tx) % {CF}
            fwd[i+1] = _partialSoftXor(fwd[i], dLlrs_ext[bx*{N_ldpc}+a]-dC2Vs[bx*{N_diags}+N_offset//2+i, tx])

        # Backward Iteration
        bwd[N_pcks-1] = np.inf
        for i in range(N_pcks-1, 0, -1):
            base   = pcks[N_offset+2*i]
            offset = pcks[N_offset+2*i+1]
            a      = base + (offset+tx) % {CF}
            bwd[i-1] = _partialSoftXor(bwd[i], dLlrs_ext[bx*{N_ldpc}+a]-dC2Vs[bx*{N_diags}+N_offset//2+i, tx])
        
        for i in range(N_pcks):
            dC2Vs[bx*{N_diags}+N_offset//2+i, tx] = _partialSoftXor(fwd[i], bwd[i])
        
        N_offset += 2*N_pcks
        
        cuda.syncthreads()

        # --------------
        # The 0-th layer
        # --------------
    
        l = N_layers-1
        N_pcks = (addr[l+1] - addr[l]) // 2

        # Forward Iteration
        for i in range(N_pcks-1):
            base   = pcks[N_offset+2*i]
            offset = pcks[N_offset+2*i+1]
            a      = base + (offset+tx) % {CF}
            fwd[i+1] = _partialSoftXor(fwd[i], dLlrs_ext[bx*{N_ldpc}+a]-dC2Vs[bx*{N_diags}+N_offset//2+i, tx])

        # Backward Iteration
        bwd[N_pcks-1] = np.inf
        for i in range(N_pcks-1, 0, -1):
            base   = pcks[N_offset+2*i]
            offset = pcks[N_offset+2*i+1]
            a      = base + (offset+tx) % {CF}
            if tx==0 and i==N_pcks-1:
                bwd[i-1] = np.inf
            else:
                bwd[i-1] = _partialSoftXor(bwd[i], dLlrs_ext[bx*{N_ldpc}+a]-dC2Vs[bx*{N_diags}+N_offset//2+i, tx])

        for i in range(N_pcks):
            if tx==0 and i==N_pcks-1:
                dC2Vs[bx*{N_diags}+N_offset//2+i, tx] = 0.0
            else:
                dC2Vs[bx*{N_diags}+N_offset//2+i, tx] = _partialSoftXor(fwd[i], bwd[i])

'''.format(ar_layerwise_pcks=str(code.llpcks.llpcks),
           ar_addr=str(code.llpcks.addr),
           N_max_num_diags=code.N_max_num_diags,
           N_ldpc=code.N,
           CF=code.nCyclicFactor,
           N_diags=code.N_diags)

        exec(s, globals())

    def update(self, dLlrs_ext, dC2Vs):
        d_update_c2v[self.N_codewords, self.code.nCyclicFactor](dLlrs_ext, dC2Vs)

        
if __name__ == '__main__':
    from sigcom.tx.modcod import LdpcEncAtsc
    import numpy as np
    from sigcom.coding.atsc import code_param_long
    import numba
    from numba import cuda

    N_codewords = 2
    CR = [8, 15]
    N_ldpc = 64800
    ldpcEncAtsc = LdpcEncAtsc(CR, N_ldpc)
    code = code_param_long.get(CR)

    Llrs_ext = np.random.randn(N_codewords*N_ldpc)
    Llrs_ext = np.asarray(Llrs_ext, np.float32)
    dLlrs_ext = cuda.to_device(Llrs_ext)

    C2Vs = np.random.randn(N_codewords*code.N_diags, code.nCyclicFactor)
    C2Vs = np.asarray(C2Vs, np.float32)
    dC2Vs = cuda.to_device(C2Vs)
    C2Vs_ref = C2Vs.copy()

    c2v = C2V(code, N_codewords)
    c2v.update(dLlrs_ext, dC2Vs)
    print(np.sum(np.abs(dC2Vs.copy_to_host()-C2Vs_ref)**2))
    #print(v2c.dLlrs_ext.copy_to_host()-Llrs)
