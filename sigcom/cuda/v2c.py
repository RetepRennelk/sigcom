import math
from numba import cuda
import numpy as np


class V2C():
    def __init__(self, code, N_codewords):
        self.code = code
        self.N_codewords = N_codewords
        
        s = '''
ar_layerwise_pcks = np.array({ar_layerwise_pcks})
ar_addr = np.array({ar_addr})

@cuda.jit
def d_update_v2c(dLlrs_ext, dLlrs, dC2Vs):
    pcks = cuda.const.array_like(ar_layerwise_pcks)
    addr = cuda.const.array_like(ar_addr)
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    
    for i in range({N_ldpc} // {CF}):
        dLlrs_ext[bx*{N_ldpc}+i*{CF}+tx] = dLlrs[bx*{N_ldpc}+i*{CF}+tx]
    
    cuda.syncthreads()

    N_layers = len(addr)-1
    N_offset = 0
    for l in range(N_layers):
        N_pcks = (addr[l+1]-addr[l]) // 2
        for i in range(N_pcks):
            base   = pcks[N_offset+2*i]
            offset = pcks[N_offset+2*i+1]
            a      = base + (offset+tx) % {CF}
            dLlrs_ext[bx*{N_ldpc}+a] += dC2Vs[bx*{N_diags}+(N_offset//2)+i, tx]
        cuda.syncthreads()

        N_offset = (N_offset + N_pcks*2) % len(pcks)

'''.format(ar_layerwise_pcks=str(code.llpcks.llpcks),
           ar_addr=str(code.llpcks.addr),
           N_ldpc=code.N,
           CF=code.nCyclicFactor,
           N_diags=code.N_diags)
        #print(s)
        exec(s, globals())

        C2Vs = np.random.randn(N_codewords*code.N_diags, code.nCyclicFactor)
        self.C2Vs = np.asarray(C2Vs, np.float32)
        self.dC2Vs = cuda.to_device(self.C2Vs)
        self.dLlrs_ext = cuda.device_array(N_codewords*code.N, dtype=np.float32)

    def update(self, dLlrs):
        d_update_v2c[self.N_codewords, self.code.nCyclicFactor](self.dLlrs_ext, dLlrs, self.dC2Vs)


if __name__ == '__main__':
    from sigcom.tx.modcod import LdpcEncAtsc
    import numpy as np
    from sigcom.coding.atsc import code_param_long
    from sigcom.rx.softxor import _partialSoftXor
    import numba
    from numba import cuda

    N_codewords = 2
    CR = [8, 15]
    N_ldpc = 64800
    ldpcEncAtsc = LdpcEncAtsc(CR, N_ldpc)
    code = code_param_long.get(CR)

    Llrs = np.random.randn(N_codewords*N_ldpc)
    Llrs = np.asarray(Llrs, np.float32)
    dLlrs = cuda.to_device(Llrs)

    v2c = V2C(code, N_codewords)
    v2c.update(dLlrs)
    print(np.sum(np.abs(v2c.dLlrs_ext.copy_to_host()-Llrs)**2))
    #print(v2c.dLlrs_ext.copy_to_host()-Llrs)
