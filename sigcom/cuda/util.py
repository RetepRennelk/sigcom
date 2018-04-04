from numba import cuda


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
