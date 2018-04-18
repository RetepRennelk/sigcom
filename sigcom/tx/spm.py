from numpy import log2, sqrt, exp, pi, array
from numpy.random import rand
from sigcom.tx.util import generate_bits, qam_alphabet, \
    map_bits_to_symbol_alphabet
from sigcom.ch.util import ricean_channel


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


class SP_rice():
    '''
    tx = sqrt(P0)*h0*tx0 + sqrt(P1)*h1*tx1
    '''
    def __init__(self, N_cells, K_factor):
        self.M = 4
        self.ldM = log2(self.M)
        self.X0 = qam_alphabet(self.M)
        self.X1 = qam_alphabet(self.M)
        self.N_cells = N_cells
        self.K_factor = K_factor
        self.update()

    def update(self):
        N_bits = int(self.ldM*self.N_cells)
        self.bits0 = generate_bits(N_bits)
        self.bits1 = generate_bits(N_bits)
        self.tx0 = map_bits_to_symbol_alphabet(self.bits0, self.X0)
        self.tx1 = map_bits_to_symbol_alphabet(self.bits1, self.X1)
        self.h0 = ricean_channel(len(self.tx0), self.K_factor)
        self.h1 = ricean_channel(len(self.tx1), self.K_factor)

    def generate(self, Ps):
        '''
        Powers : Ps =  [P0, P1]
        '''
        self.Ps = array(Ps)
        self.tx = sqrt(Ps[0])*self.h0*self.tx0 + sqrt(Ps[1])*self.h1*self.tx1
