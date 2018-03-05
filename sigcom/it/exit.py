import numpy as np
from sigcom.coding.atsc import code_param_long
from sigcom.coding.atsc import code_param_short
from sigcom.coding.PCM import PCM


class EXIT_mod_vn():
    '''
    Combined EXIT function of demapper and variable nodes
    '''
    def __init__(self, M, CR, N_ldpc):
        self.M = M
        self.ldM = int(np.log2(M))
        self.CR = CR
        self.N_ldpc = N_ldpc
        if N_ldpc == 16200:
            self.cp = code_param_short.get(CR)
        elif N_ldpc == 64800:
            self.cp = code_param_long.get(CR)
        self.pcm = PCM(self.cp)

