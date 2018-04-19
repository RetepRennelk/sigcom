import numba
from numba import cuda
import numpy as np


s_update_v2c = '''
ar_layerwise_pcks = np.array({ar_layerwise_pcks})
ar_addr = np.array({ar_addr})

@cuda.jit(device=True)
def d_update_v2c(dLlrs_ext, dLlrs, dC2Vs):
    pcks = cuda.const.array_like(ar_layerwise_pcks)
    addr = cuda.const.array_like(ar_addr)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    for i in range({N_ldpc} // {CF}):
        dLlrs_ext[bx*{N_ldpc}+i*{CF}+tx] = dLlrs[bx*{N_ldpc}+i*{CF}+tx]

    cuda.syncthreads()

    # Process layers 1..N_layers-1

    N_layers = len(addr)-1
    N_offset = addr[1]
    for l in range(1, N_layers):
        N_pcks = (addr[l+1]-addr[l]) // 2
        for i in range(N_pcks):
            base   = pcks[N_offset+2*i]
            offset = pcks[N_offset+2*i+1]
            a      = base + (offset+tx) % {CF}
            dLlrs_ext[bx*{N_ldpc}+a] += dC2Vs[bx*{N_diags}+(N_offset//2)+i, tx]
        cuda.syncthreads()
        N_offset = (N_offset + N_pcks*2) % len(pcks)

    # The 0-th layer

    l = 0
    N_pcks = addr[1] // 2
    for i in range(N_pcks):
        base   = pcks[2*i]
        offset = pcks[2*i+1]
        a      = base + (offset+tx) % {CF}
        dLlrs_ext[bx*{N_ldpc}+a] += dC2Vs[bx*{N_diags}+i, tx]
'''


@cuda.jit
def decode(dLlrs_ext, dLlrs, dC2Vs, nMaxFecIterations, N_diags, clip_level):
    d_update_v2c(dLlrs_ext, dLlrs, dC2Vs)


class GpuFecDec():
    def __init__(self, code, N_codewords, clip_level=16):
        self.code = code
        self.N_codewords = N_codewords
        self.clip_level = clip_level
        CF = code.nCyclicFactor
        # The diagonals are stored 'vertically', one below the other
        self.dC2Vs = cuda.device_array((N_codewords*code.N_diags, CF), np.float32)
        self.dLlrs_ext = cuda.device_array(N_codewords*code.N, np.float32)

        exec(s_update_v2c.format(ar_layerwise_pcks=str(code.llpcks.llpcks),
                                 ar_addr=str(code.llpcks.addr),
                                 N_ldpc=code.N,
                                 CF=code.nCyclicFactor,
                                 N_diags=code.N_diags), globals())

    def decode(self, dLlrs, nMaxFecIterations):
        CF = self.code.nCyclicFactor
        decode[self.N_codewords, CF](self.dLlrs_ext, dLlrs,
                                     self.dC2Vs, 
                                     nMaxFecIterations, self.code.N_diags,
                                     self.clip_level)
