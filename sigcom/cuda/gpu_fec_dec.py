import numba
from numba import cuda
import numpy as np
from sigcom.cuda.util import _partialSoftXor
from numba.cuda.cudadrv.driver import device_memset


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

    if 1:
        # Process layers 1..N_layers-1

        N_layers = len(addr)-1
        N_offset = addr[1]
        for l in range(1,N_layers):
            N_pcks = (addr[l+1]-addr[l]) // 2
            for i in range(N_pcks):
                base   = pcks[N_offset+2*i]
                offset = pcks[N_offset+2*i+1]
                a      = base + (offset+tx) % {CF}
                dLlrs_ext[bx*{N_ldpc}+a] += dC2Vs[bx*{N_diags}+(N_offset//2)+i, tx]
            cuda.syncthreads()
            N_offset = (N_offset + N_pcks*2) % len(pcks)

    if 1:
        # The 0-th layer
        l = 0
        N_pcks = addr[1] // 2
        for i in range(N_pcks):
            base   = pcks[2*i]
            offset = pcks[2*i+1]
            a      = base + (offset+tx) % {CF}
            dLlrs_ext[bx*{N_ldpc}+a] += dC2Vs[bx*{N_diags}+i, tx]
'''

s_update_c2v = '''
@cuda.jit(device=True)
def d_update_c2v(dLlrs_ext, dC2Vs):
    pcks = cuda.const.array_like(ar_layerwise_pcks)
    addr = cuda.const.array_like(ar_addr)
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x

    fwd = cuda.local.array({N_max_num_diags}, dtype=numba.float32)
    bwd = cuda.local.array({N_max_num_diags}, dtype=numba.float32)       

    fwd[0] = np.inf

    # Layers 1..N_layers-1
    
    N_layers = len(addr)-1
    N_offset = addr[1]
    for l in range(1, N_layers):
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

    # The 0-th layer
    
    l = 0
    N_pcks = addr[1] // 2

    # Forward Iteration
    for i in range(N_pcks-1):
        base   = pcks[2*i]
        offset = pcks[2*i+1]
        a      = base + (offset + tx) % {CF}
        fwd[i+1] = _partialSoftXor(fwd[i], dLlrs_ext[bx*{N_ldpc}+a]-dC2Vs[bx*{N_diags}+i, tx])

    # Backward Iteration
    bwd[N_pcks-1] = np.inf
    for i in range(N_pcks-1, 0, -1):
        base   = pcks[2*i]
        offset = pcks[2*i+1]
        a      = base + (offset + tx) % {CF}
        if tx==0 and i==N_pcks-1:
            bwd[i-1] = np.inf
        else:
            bwd[i-1] = _partialSoftXor(bwd[i], dLlrs_ext[bx*{N_ldpc}+a]-dC2Vs[bx*{N_diags}+i, tx])

    for i in range(N_pcks):
        if tx==0 and i==N_pcks-1:
            dC2Vs[bx*{N_diags}+i, tx] = 0.0
        else:
            dC2Vs[bx*{N_diags}+i, tx] = _partialSoftXor(fwd[i], bwd[i])
'''

s_check_codeword = '''
@cuda.jit(device=True)
def d_check_codeword(dLlrs):
    pcks = cuda.const.array_like(ar_layerwise_pcks)
    addr = cuda.const.array_like(ar_addr)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x

    buf = cuda.shared.array({CF}, dtype=numba.int64)

    N_offset = addr[1]
    N_layers = len(addr)-1
    
    buf[tx] = 0
    cuda.syncthreads()

    for l in range(1, N_layers):
        N_pcks = (addr[l+1] - addr[l]) // 2
        S = 0
        for i in range(N_pcks):
            base   = pcks[N_offset+2*i]
            offset = pcks[N_offset+2*i+1]
            a      = base + (offset + tx) % {CF}
            S += np.int64(dLlrs[bx*{N_ldpc}+a] < 0)
        buf[tx] += S % 2
        cuda.syncthreads()
        N_offset = N_offset + N_pcks*2
    
    l = 0
    N_pcks = addr[1] // 2
    S = 0
    for i in range(N_pcks):
        base   = pcks[2*i]
        offset = pcks[2*i+1]
        a      = base + (offset + tx) % {CF}
        if i < N_pcks-1 and tx == 0:
            S += np.int64(dLlrs[bx*{N_ldpc}+a] < 0)
    buf[tx] += S % 2
    cuda.syncthreads()
    
    N_steps = {CF} // 2
    while N_steps > 0:
        if tx < N_steps:
            buf[tx] += buf[tx+N_steps]
        cuda.syncthreads()
        if N_steps % 2 == 0:
            N_steps //= 2
        else:
            break
    if tx == 0:
        for i in range(1, N_steps):
            buf[0] += buf[i]
    cuda.syncthreads()
    
    return buf[0]
'''

@cuda.jit(device=True)
def clip(dC2Vs, clip_level, N_diags):
    bx = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    for d in range(N_diags):
        level = dC2Vs[bx*N_diags+d,tx]
        if level < -clip_level:
            level = -clip_level
        elif level > clip_level:
            level = clip_level
        dC2Vs[bx*N_diags+d,tx] = level
        cuda.syncthreads()

@cuda.jit
def decode(dLlrs_ext, dLlrs, dC2Vs, dIterations, nMaxFecIterations, N_diags, clip_level):
    bx = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    for it in range(nMaxFecIterations+1):
        d_update_v2c(dLlrs_ext, dLlrs, dC2Vs)
        clip(dC2Vs, clip_level, N_diags)
        if d_check_codeword(dLlrs_ext) == 0:
            if tx == 0:
                dIterations[bx] = it
            return
        d_update_c2v(dLlrs_ext, dC2Vs)
        clip(dC2Vs, clip_level, N_diags)
    if tx == 0:
        dIterations[bx] = it

class GpuFecDec():
    def __init__(self, code, N_codewords, clip_level=16):
        self.code = code
        self.N_codewords = N_codewords
        self.clip_level = clip_level
        CF = code.nCyclicFactor
        # The diagonals are stored 'vertically', one below the other
        self.dC2Vs = cuda.device_array((N_codewords*code.N_diags, CF), np.float32)
        self.dLlrs_ext = cuda.device_array(N_codewords*code.N, np.float32)
        self.dIterations = cuda.device_array(N_codewords, np.int32)
        
        exec(s_update_v2c.format(ar_layerwise_pcks=str(code.llpcks.llpcks),
                                 ar_addr=str(code.llpcks.addr),
                                 N_ldpc=code.N,
                                 CF=code.nCyclicFactor,
                                 N_diags=code.N_diags), globals())

        exec(s_update_c2v.format(N_max_num_diags=code.N_max_num_diags,
                                 N_ldpc=code.N,
                                 CF=code.nCyclicFactor,
                                 N_diags=code.N_diags), globals())

        exec(s_check_codeword.format(CF=code.nCyclicFactor,
                                     N_ldpc=code.N), globals())

    def reset_C2Vs(self):
        N_bytes = 4*self.code.N_diags*self.code.nCyclicFactor*self.N_codewords
        device_memset(self.dC2Vs, 0, N_bytes)
        
    def decode(self, dLlrs, nMaxFecIterations):
        self.reset_C2Vs()
        CF = self.code.nCyclicFactor
        decode[self.N_codewords, CF](self.dLlrs_ext, dLlrs,
                                     self.dC2Vs, self.dIterations,
                                     nMaxFecIterations, self.code.N_diags,
                                     self.clip_level)
