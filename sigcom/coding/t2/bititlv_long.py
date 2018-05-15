import numpy as np
from sigcom.coding.t2.column_twist_interleaver import ColumnTwistInterleaver
from sigcom.coding.t2.bit_to_cell_demux import bit_to_cell_demux

def bititlv_long(M, CR):
    N_ldpc = 64800
    ctil = ColumnTwistInterleaver(M, N_ldpc)
    bil = ctil.getPermVector()
    b2c = bit_to_cell_demux(M, N_ldpc, CR)
    N_b2c = len(b2c)
    x, y = np.meshgrid(b2c, np.arange(N_ldpc//N_b2c)*N_b2c)
    idx = (x+y).flatten()
    return bil[idx]
    
    
