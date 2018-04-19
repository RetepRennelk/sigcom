import numba
from numba import cuda
import numpy as np
from sigcom.cuda.util import _partialSoftXor
from numba.cuda.cudadrv.driver import device_memset


s_update_v2c = '''
ar_layerwise_pcks = np.array({ar_layerwise_pcks})
ar_addr = np.array({ar_addr})

@cuda.jit('void(float32[:],float32[:],float32[:,:])', device=True)
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

s_update_v2c_unroll = '''
@cuda.jit('void(float32[:],float32[:],float32[:,:])', device=True)
def d_update_v2c_unroll(dLlrs_ext, dLlrs, dC2Vs):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    for i in range({N_ldpc} // {CF}):
        dLlrs_ext[bx*{N_ldpc}+i*{CF}+tx] = dLlrs[bx*{N_ldpc}+i*{CF}+tx]

    cuda.syncthreads()
'''

@cuda.jit
def decode(dLlrs_ext, dLlrs, dC2Vs, nMaxFecIterations, N_diags):
    for it in range(nMaxFecIterations+1):
        d_update_v2c_unroll(dLlrs_ext, dLlrs, dC2Vs)


class GpuFecDec():
    def __init__(self, code, N_codewords, clip_level=16):
        self.code = code
        self.N_codewords = N_codewords
        self.clip_level = clip_level
        CF = code.nCyclicFactor
        # The diagonals are stored 'vertically', one below the other
        self.dC2Vs = cuda.device_array((N_codewords*code.N_diags, CF), np.float32)
        self.dLlrs_ext = cuda.device_array(N_codewords*code.N, np.float32)

        s = s_update_v2c_unroll.format(N_ldpc=code.N,
                                       CF=CF)
        pcks = code.llpcks.llpcks
        addr = code.llpcks.addr
        N_layers = len(code.llpcks.addr)-1
        N_offset = code.llpcks.addr[1]
        for l in range(1, N_layers):
            N_pcks = (addr[l+1]-addr[l]) // 2
            for i in range(N_pcks):
                base = pcks[N_offset+2*i]
                offset = pcks[N_offset+2*i+1]
                # a = base + (offset+tx) % CF
                # dLlrs_ext[bx*{N_ldpc}+{a}] += dC2Vs[bx*{N_diags}+(N_offset//2)+i, tx]
                s +='''
    a = {base} + ({offset}+tx) % {CF}
    dLlrs_ext[bx*{N_ldpc}+a] += dC2Vs[bx*{N_diags}+{N_offset_2}+{i}, tx]
'''.format(base=base,
           offset=offset,
           CF=CF,
           N_ldpc=code.N,
           i=i,
           N_diags=code.N_diags,
           N_offset_2=N_offset//2)

            s += '    cuda.syncthreads()\n'
            N_offset = N_offset + N_pcks*2

        N_pcks = addr[1] // 2
        for i in range(N_pcks):
            base = pcks[2*i]
            offset = pcks[2*i+1]
            # a = base + (offset+tx) % CF
            # dLlrs_ext[bx*{N_ldpc}+a] += dC2Vs[bx*{N_diags}+i, tx]
            s +='''
    a = {base} + ({offset}+tx) % {CF}
    dLlrs_ext[bx*{N_ldpc}+a] += dC2Vs[bx*{N_diags}+{i}, tx]
'''.format(CF=CF,
           base=base,
           offset=offset,
           N_ldpc=code.N,
           i=i,
           N_diags=code.N_diags)
        exec(s, globals())

    def decode(self, dLlrs, nMaxFecIterations):
        CF = self.code.nCyclicFactor
        decode[self.N_codewords, CF](self.dLlrs_ext, dLlrs,
                                     self.dC2Vs,
                                     nMaxFecIterations,
                                     self.code.N_diags)


if __name__ == '__main__':
    from sigcom.coding.atsc import code_param_long
    from sigcom.tx.modcod import LdpcEncAtsc

    N_codewords = 1
    CR = [8, 15]
    N_ldpc = 64800
    code = code_param_long.get(CR)
    ldpcEncAtsc = LdpcEncAtsc(CR, N_ldpc)
    gpuFecDec = GpuFecDec(code, N_codewords)

    ldpcEncAtsc.generate(N_codewords)
    tx = 1-2*ldpcEncAtsc.codebits
    noise = np.random.randn(len(tx))
    P_noise = .5
    rx = tx+noise*np.sqrt(P_noise)
    Llrs = np.asarray(2/P_noise*rx, np.float32)
    dLlrs = cuda.to_device(Llrs)

    CF = code.nCyclicFactor
    N_diags = code.N_diags
    C2Vs = np.random.randn(N_diags*N_codewords, CF)
    C2Vs = np.asarray(C2Vs, dtype=np.float32)
    dC2Vs = cuda.to_device(C2Vs)
    gpuFecDec.dC2Vs = dC2Vs

    N_max_fec_iterations = 1
    gpuFecDec.decode(dLlrs, N_max_fec_iterations)
