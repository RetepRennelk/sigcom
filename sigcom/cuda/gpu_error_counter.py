import numba
from numba import cuda
import numpy as np
from sigcom.cuda.util import Gpu


s_count_frame_errors = '''
@cuda.jit
def d_count_frame_errors(dFrameErrors, dLlrs, dBits):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    gw = cuda.gridDim.x

    buf = cuda.shared.array({CF}, dtype=numba.int64)

    while bx < {N_codewords}:
        S = 0
        for k in range({K} // {CF}):
            a = bx*{N_ldpc} + k*{CF} + tx
            S += np.int64((dLlrs[a]<0.0) != dBits[a])
        buf[tx] = S
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
            dFrameErrors[bx] = buf[0]
        cuda.syncthreads()

        bx += gw
'''


class GpuErrorCounter():
    '''
    Assumes a CR=K/N LDPC code and counts the errors per frame within the
    systematic information part 0..K-1
    '''
    def __init__(self, code, N_codewords):
        self.code = code
        self.N_codewords = N_codewords
        self.dFrameErrors = cuda.device_array(N_codewords, np.int64)
        self.gpu = Gpu()
        exec(s_count_frame_errors.format(CF=code.nCyclicFactor,
                                         N_ldpc=code.N,
                                         K=code.K,
                                         N_codewords=N_codewords), globals())

    def count_frame_errors(self, dLlrs, dBits):
        B = 2*self.gpu.MULTIPROCESSOR_COUNT
        T = self.code.nCyclicFactor
        d_count_frame_errors[B, T](self.dFrameErrors, dLlrs, dBits)
