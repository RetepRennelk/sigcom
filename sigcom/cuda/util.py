import math
from numba import cuda


@cuda.jit(device=True, inline=True)
def _fmin(x, y):
    if x < y:
        return x
    else:
        return y


@cuda.jit(device=True, inline=True)
def _sign(x):
    if x < 0:
        return -1.0
    else:
        return 1.0


@cuda.jit(device=True, inline=True)
def _correctionTerm(x):
    return math.log(1.0+math.exp(-x))


@cuda.jit(device=True, inline=True)
def _partialSoftXor(L1, L2):
    L1_abs = math.fabs(L1)
    L2_abs = math.fabs(L2)
    rhs0 = _fmin(L1_abs, L2_abs)
    rhs1 = _correctionTerm(L1_abs+L2_abs)
    rhs2 = _correctionTerm(math.fabs(L1_abs-L2_abs))
    return _sign(L1)*_sign(L2)*(rhs0+rhs1-rhs2)


def print_attr_current_device():
    gpu = cuda.get_current_device()
    print("name = %s" % gpu.name)
    print("maxThreadsPerBlock = %s" % str(gpu.MAX_THREADS_PER_BLOCK))
    print("maxBlockDimX = %s" % str(gpu.MAX_BLOCK_DIM_X))
    print("maxBlockDimY = %s" % str(gpu.MAX_BLOCK_DIM_Y))
    print("maxBlockDimZ = %s" % str(gpu.MAX_BLOCK_DIM_Z))
    print("maxGridDimX = %s" % str(gpu.MAX_GRID_DIM_X))
    print("maxGridDimY = %s" % str(gpu.MAX_GRID_DIM_Y))
    print("maxGridDimZ = %s" % str(gpu.MAX_GRID_DIM_Z))
    print("maxSharedMemoryPerBlock = %s" % str(gpu.MAX_SHARED_MEMORY_PER_BLOCK))
    print("asyncEngineCount = %s" % str(gpu.ASYNC_ENGINE_COUNT))
    print("canMapHostMemory = %s" % str(gpu.CAN_MAP_HOST_MEMORY))
    print("multiProcessorCount = %s" % str(gpu.MULTIPROCESSOR_COUNT))
    print("warpSize = %s" % str(gpu.WARP_SIZE))
    print("unifiedAddressing = %s" % str(gpu.UNIFIED_ADDRESSING))
    print("pciBusID = %s" % str(gpu.PCI_BUS_ID))
    print("pciDeviceID = %s" % str(gpu.PCI_DEVICE_ID))


class Gpu():
    def __init__(self):
        gpu = cuda.get_current_device()
        self.name = gpu.name
        self.MAX_THREADS_PER_BLOCK = gpu.MAX_THREADS_PER_BLOCK
        self.MAX_BLOCK_DIM_X = gpu.MAX_BLOCK_DIM_X
        self.MAX_BLOCK_DIM_Y = gpu.MAX_BLOCK_DIM_Y
        self.MAX_BLOCK_DIM_Z = gpu.MAX_BLOCK_DIM_Z
        self.MAX_GRID_DIM_X = gpu.MAX_GRID_DIM_X
        self.MAX_GRID_DIM_Y = gpu.MAX_GRID_DIM_Y
        self.MAX_GRID_DIM_Z = gpu.MAX_GRID_DIM_Z
        self.MAX_SHARED_MEMORY_PER_BLOCK = gpu.MAX_SHARED_MEMORY_PER_BLOCK
        self.ASYNC_ENGINE_COUNT = gpu.ASYNC_ENGINE_COUNT
        self.CAN_MAP_HOST_MEMORY = gpu.CAN_MAP_HOST_MEMORY
        self.MULTIPROCESSOR_COUNT = gpu.MULTIPROCESSOR_COUNT
        self.WARP_SIZE = gpu.WARP_SIZE
        self.UNIFIED_ADDRESSING = gpu.UNIFIED_ADDRESSING
        self.PCI_BUS_ID = gpu.PCI_BUS_ID
        self.PCI_DEVICE_ID = gpu.PCI_DEVICE_ID
