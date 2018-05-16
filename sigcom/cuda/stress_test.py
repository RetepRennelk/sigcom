import numpy as np
from numba import cuda


@cuda.jit
def func(y, x):
    tid = cuda.grid(1)
    bw = cuda.gridsize(1)
    while tid < len(x):
        y[tid] = x[tid]
        tid += bw


N_threads = 256
Its = 100
for it in range(Its):
    for i in range(1, 25):
        N = 2**i
        B = int(np.ceil((N+1)/N_threads))
        x = np.asarray(np.random.randn(N), dtype=np.float32)
        y = np.zeros(N, dtype=np.float32)
        func[B, N_threads](y, x)
        assert np.sum((y-x)**2) < 1e-10
        print('N={}, it={} okay'.format(N, it))
