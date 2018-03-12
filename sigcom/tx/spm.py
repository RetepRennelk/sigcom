from numpy import log2, sqrt, exp, pi, array
from numpy.random import rand
from sigcom.tx.util import generate_bits, qam_alphabet, \
    map_bits_to_symbol_alphabet
from sigcom.tx.mod_cod_atsc import ModCodAtsc
import numpy as np


class SP1p4():
    '''
    tx = sqrt(P0)*tx0 + sqrt(P1)*tx1*exp(j*phi)
    '''
    def __init__(self, N_cells):
        self.M = 4
        self.ldM = log2(self.M)
        self.X0 = qam_alphabet(self.M)
        self.X1 = qam_alphabet(self.M)
        self.N_cells = N_cells
        self.update()

    def update(self):
        N_bits = int(self.ldM*self.N_cells)
        self.bits0 = generate_bits(N_bits)
        self.bits1 = generate_bits(N_bits)
        self.tx0 = map_bits_to_symbol_alphabet(self.bits0, self.X0)
        self.tx1 = map_bits_to_symbol_alphabet(self.bits1, self.X1)
        self.phase = exp(1j*2*pi*rand(self.N_cells))

    def generate(self, Ps):
        '''
        Powers : Ps =  [P0, P1]
        '''
        self.Ps = array(Ps)
        self.tx = sqrt(Ps[0])*self.tx0 + sqrt(Ps[1])*self.tx1*self.phase


class ModCodSP1p4():
    def __init__(self, M, CR, N_ldpc):
        self.tx0 = ModCodAtsc(M, CR, N_ldpc)
        self.tx1 = ModCodAtsc(M, CR, N_ldpc)

    def update(self, N_codewords):
        self.tx0.generate(N_codewords)
        self.tx1.generate(N_codewords)
        N_cells = len(self.tx0.tx)
        self.phase = exp(1j*2*pi*rand(N_cells))

    def generate(self, Ps):
        self.tx = np.sqrt(Ps[0])*self.tx0.tx + np.sqrt(Ps[1])*self.tx1.tx*self.phase


class ModCodSP1p4_rs():
    def __init__(self, M, CR, N_ldpc):
        self.N_fec_cells = int(N_ldpc / np.log2(M))
        self.tx0 = ModCodAtsc(M, CR, N_ldpc)
        self.tx1 = ModCodAtsc(M, CR, N_ldpc)

    def update(self, N_codewords):
        self.tx0.generate(N_codewords)
        self.tx1.generate(N_codewords)
        N_cells = len(self.tx0.tx)
        self.phase = exp(1j*2*pi*rand(N_cells))

    def generate(self, Ps):
        #import ipdb; ipdb.set_trace()
        N_fec_cells = self.N_fec_cells
        m_tx0 = self.tx0.tx.reshape(-1, N_fec_cells).copy()
        m_tx1 = self.tx1.tx.reshape(-1, N_fec_cells).copy()
        m_tx0[:,:int(N_fec_cells/2)] *= np.sqrt(Ps[0])
        m_tx1[:,:int(N_fec_cells/2)] *= np.sqrt(Ps[1])
        m_tx0[:,int(N_fec_cells/2):] *= np.sqrt(Ps[1])
        m_tx1[:,int(N_fec_cells/2):] *= np.sqrt(Ps[0])
        self.tx = m_tx0.flatten() + m_tx1.flatten()*self.phase
