import math
import cmath
import numba
import numpy as np
from numba import cuda
from sigcom.tx.util import make_cells
from sigcom.ch.util import make_noise
from sigcom.tx.util import qam_alphabet
import matplotlib.pyplot as plt


@cuda.jit(device=True)
def max_star(a,b):
    return max((a,b))+math.log(1.+math.exp(-abs(a-b)))


def qam2str(qam):
    sqam = ','.join([str(q) for q in qam])
    return '[' + sqam + ']'


sMI0='''
cqam0 = np.array({sqam0})
cqam1 = np.array({sqam1})

@cuda.jit
def gMI0(MIs, tx0, tx1, noise, h0, h1, C_Ns_dB, C_Is_dB, rho_dB):
    qam0 = cuda.const.array_like(cqam0)
    qam1 = cuda.const.array_like(cqam1)
    
    bidx = cuda.blockIdx.x
    bidy = cuda.blockIdx.y
    
    C_I = 10.**(C_Is_dB[bidy]/10.)
    C_N = 10.**(C_Ns_dB[bidx]/10.)
    rho = 10.**(rho_dB/10)

    Cp_sqrt  = math.sqrt(2.*rho/(1.+rho))
    Cpp_sqrt = math.sqrt(2./(1.+rho))
    Ip_sqrt  = math.sqrt(2./(1.+rho)/C_I)
    Ipp_sqrt = math.sqrt(2.*rho/(1.+rho)/C_I)
    
    P_noise = 1./C_N
    
    N_cells = len(tx0)
    MI0 = 0.
    for k in range(N_cells):
        if (k//(N_cells//2)) % 2 == 0:
            C_sqrt = Cp_sqrt
            I_sqrt = Ip_sqrt
        else:
            C_sqrt = Cpp_sqrt
            I_sqrt = Ipp_sqrt
        
        rx = C_sqrt*tx0[k]*h0[k] + I_sqrt*tx1[k]*h1[k] + noise[k]*math.sqrt(P_noise)
        
        num = -np.inf
        den = -np.inf
        for x1 in qam1:
            D = rx - C_sqrt*h0[k]*tx0[k] - I_sqrt*x1*h1[k]
            num = max_star(num, -abs(D)**2/P_noise)
            for x0 in qam0:
                D = rx - C_sqrt*x0*h0[k] - I_sqrt*x1*h1[k]
                den = max_star(den, -abs(D)**2/P_noise)
        MI0 += num - den
    MIs[bidy, bidx] = {ldM0} + MI0/N_cells/math.log(2.0)
'''

sMI1='''
cqam0 = np.array({sqam0})
cqam1 = np.array({sqam1})

@cuda.jit
def gMI1(MIs, tx0, tx1, noise, h0, h1, C_Ns_dB, C_Is_dB, rho_dB):
    qam0 = cuda.const.array_like(cqam0)
    qam1 = cuda.const.array_like(cqam1)
    
    bidx = cuda.blockIdx.x
    bidy = cuda.blockIdx.y
    
    C_I = 10.**(C_Is_dB[bidy]/10.)
    C_N = 10.**(C_Ns_dB[bidx]/10.)
    rho = 10.**(rho_dB/10)

    Cp_sqrt  = math.sqrt(2.*rho/(1.+rho))
    Cpp_sqrt = math.sqrt(2./(1.+rho))
    Ip_sqrt  = math.sqrt(2./(1.+rho)/C_I)
    Ipp_sqrt = math.sqrt(2.*rho/(1.+rho)/C_I)
    
    P_noise = 1./C_N
    
    N_cells = len(tx0)
    MI1 = 0.
    for k in range(N_cells):
        if (k//(N_cells//2)) % 2 == 0:
            C_sqrt = Cp_sqrt
            I_sqrt = Ip_sqrt
        else:
            C_sqrt = Cpp_sqrt
            I_sqrt = Ipp_sqrt
        
        rx = C_sqrt*tx0[k]*h0[k] + I_sqrt*tx1[k]*h1[k] + noise[k]*math.sqrt(P_noise)
        
        num = -np.inf
        den = -np.inf
        for x0 in qam0:
            D = rx - C_sqrt*h0[k]*x0 - I_sqrt*tx1[k]*h1[k]
            num = max_star(num, -abs(D)**2/P_noise)
            for x1 in qam1:
                D = rx - C_sqrt*x0*h0[k] - I_sqrt*x1*h1[k]
                den = max_star(den, -abs(D)**2/P_noise)
        MI1 += num - den
    MIs[bidy, bidx] = {ldM1} + MI1/N_cells/math.log(2.0)
'''


sMI1_0='''
cqam0 = np.array({sqam0})
cqam1 = np.array({sqam1})

@cuda.jit
def gMI1_0(MIs, tx0, tx1, noise, h0, h1, C_Ns_dB, C_Is_dB, rho_dB):
    qam0 = cuda.const.array_like(cqam0)
    qam1 = cuda.const.array_like(cqam1)
    
    bidx = cuda.blockIdx.x
    bidy = cuda.blockIdx.y
    
    C_I = 10.**(C_Is_dB[bidy]/10.)
    C_N = 10.**(C_Ns_dB[bidx]/10.)
    rho = 10.**(rho_dB/10)

    Cp_sqrt  = math.sqrt(2.*rho/(1.+rho))
    Cpp_sqrt = math.sqrt(2./(1.+rho))
    Ip_sqrt  = math.sqrt(2./(1.+rho)/C_I)
    Ipp_sqrt = math.sqrt(2.*rho/(1.+rho)/C_I)
    
    P_noise = 1./C_N
    
    N_cells = len(tx0)
    MI1_0 = 0.
    for k in range(N_cells):
        if (k//(N_cells//2)) % 2 == 0:
            C_sqrt = Cp_sqrt
            I_sqrt = Ip_sqrt
        else:
            C_sqrt = Cpp_sqrt
            I_sqrt = Ipp_sqrt
        
        rx = C_sqrt*tx0[k]*h0[k] + I_sqrt*tx1[k]*h1[k] + noise[k]*math.sqrt(P_noise)
        
        D = rx - C_sqrt*tx0[k]*h0[k] - I_sqrt*tx1[k]*h1[k]
        num = -abs(D)**2/P_noise
        
        den = -np.inf
        for x1 in qam1:
            D = rx - C_sqrt*tx0[k]*h0[k] - I_sqrt*x1*h1[k]
            den = max_star(den, -abs(D)**2/P_noise)
            
        MI1_0 += num - den
    MIs[bidy, bidx] = {ldM1} + MI1_0/N_cells/math.log(2.0)
'''

sMI0_1='''
cqam0 = np.array({sqam0})
cqam1 = np.array({sqam1})

@cuda.jit
def gMI0_1(MIs, tx0, tx1, noise, h0, h1, C_Ns_dB, C_Is_dB, rho_dB):
    qam0 = cuda.const.array_like(cqam0)
    qam1 = cuda.const.array_like(cqam1)
    
    bidx = cuda.blockIdx.x
    bidy = cuda.blockIdx.y
    
    C_I = 10.**(C_Is_dB[bidy]/10.)
    C_N = 10.**(C_Ns_dB[bidx]/10.)
    rho = 10.**(rho_dB/10)

    Cp_sqrt  = math.sqrt(2.*rho/(1.+rho))
    Cpp_sqrt = math.sqrt(2./(1.+rho))
    Ip_sqrt  = math.sqrt(2./(1.+rho)/C_I)
    Ipp_sqrt = math.sqrt(2.*rho/(1.+rho)/C_I)
    
    P_noise = 1./C_N
    
    N_cells = len(tx0)
    MI0_1 = 0.
    for k in range(N_cells):
        if (k//(N_cells//2)) % 2 == 0:
            C_sqrt = Cp_sqrt
            I_sqrt = Ip_sqrt
        else:
            C_sqrt = Cpp_sqrt
            I_sqrt = Ipp_sqrt
        
        rx = C_sqrt*tx0[k]*h0[k] + I_sqrt*tx1[k]*h1[k] + noise[k]*math.sqrt(P_noise)
        
        D = rx - C_sqrt*tx0[k]*h0[k] - I_sqrt*tx1[k]*h1[k]
        num = -abs(D)**2/P_noise
        
        den = -np.inf
        for x0 in qam0:
            D = rx - C_sqrt*x0*h0[k] - I_sqrt*tx1[k]*h1[k]
            den = max_star(den, -abs(D)**2/P_noise)
            
        MI0_1 += num - den
    MIs[bidy, bidx] = {ldM0} + MI0_1/N_cells/math.log(2.0)
'''

@cuda.jit
def gMI_sum(MIs_sum, MI0s, MI1_0s, MI1s, MI0_1s):
    bidx = cuda.blockIdx.x
    bidy = cuda.blockIdx.y
    MIs_sum[bidy, bidx] = 0.5*(MI0s[bidy, bidx] + MI1_0s[bidy, bidx] + MI1s[bidy, bidx] + MI0_1s[bidy, bidx])


@cuda.jit
def gMI_max_min(Rs, MIs_sum, MI0s, MI0_1s):
    bidx = cuda.blockIdx.x
    bidy = cuda.blockIdx.y
    R = min((0.5*MIs_sum[bidy, bidx], MI0_1s[bidy, bidx]))
    Rs[bidy, bidx] = max((R, MI0s[bidy, bidx]))


class MI_SP1p_rs_ofdm():
    def __init__(self, M0, M1, N_cells):
        self.qam0 = qam_alphabet(M0)
        self.qam1 = qam_alphabet(M0)
        sqam0 = qam2str(self.qam0)
        sqam1 = qam2str(self.qam1)
        self.N_cells = N_cells
        self.update()

        s = sMI0.format(sqam0=sqam0, sqam1=sqam1, 
                        M0=M0, ldM0=int(np.log2(M0)),
                        M1=M0, ldM1=int(np.log2(M1)))
        exec(s, globals())

        s = sMI1.format(sqam0=sqam0, sqam1=sqam1, 
                        M0=M0, ldM0=int(np.log2(M0)),
                        M1=M1, ldM1=int(np.log2(M1)))
        exec(s, globals())

        s = sMI1_0.format(sqam0=sqam0, sqam1=sqam1, 
                          M0=M0, ldM0=int(np.log2(M0)),
                          M1=M1, ldM1=int(np.log2(M1)))
        exec(s, globals())

        s = sMI0_1.format(sqam0=sqam0, sqam1=sqam1, 
                          M0=M0, ldM0=int(np.log2(M0)),
                          M1=M1, ldM1=int(np.log2(M1)))
        exec(s, globals())
       

    def update(self):
        tx0, bits0 = make_cells(self.qam0, self.N_cells)
        tx1, bits1 = make_cells(self.qam1, self.N_cells)
        noise = make_noise(self.N_cells)
        phase = np.exp(1j*2*np.pi*np.random.rand(self.N_cells))
        self.tx0 = np.asarray(tx0, np.complex64)
        self.tx1 = np.asarray(tx1, np.complex64)
        self.noise = np.asarray(noise, np.complex64)
        self.phase = np.asarray(phase, np.complex64)

    def compute_MIs(self, C_Is_dB, C_Ns_dB, rho_dB):
        h0 = np.ones(self.N_cells, dtype=np.complex64)
        h1 = self.phase

        self.C_Is_dB = C_Is_dB
        self.C_Ns_dB = C_Ns_dB
        self.rho_dB = rho_dB
        
        self.MI0s = np.zeros((len(C_Is_dB), len(C_Ns_dB)), dtype=np.float32)
        gMI0[(len(C_Is_dB), len(C_Ns_dB)),1](self.MI0s, self.tx0, self.tx1,
                                             self.noise,
                                             h0, h1,
                                             C_Ns_dB, C_Is_dB, rho_dB)

        self.MI1s = np.zeros((len(C_Is_dB), len(C_Ns_dB)), dtype=np.float32)
        gMI1[(len(C_Is_dB), len(C_Ns_dB)),1](self.MI1s, self.tx0, self.tx1,
                                             self.noise,
                                             h0, h1,
                                             C_Ns_dB, C_Is_dB, rho_dB)

        self.MI1_0s = np.zeros((len(C_Is_dB), len(C_Ns_dB)), dtype=np.float32)
        gMI1_0[(len(C_Is_dB), len(C_Ns_dB)),1](self.MI1_0s, self.tx0, self.tx1,
                                               self.noise,
                                               h0, h1,
                                               C_Ns_dB, C_Is_dB, rho_dB)
        
        self.MI0_1s = np.zeros((len(C_Is_dB), len(C_Ns_dB)), dtype=np.float32)
        gMI0_1[(len(C_Is_dB), len(C_Ns_dB)),1](self.MI0_1s, self.tx0, self.tx1,
                                               self.noise,
                                               h0, h1,
                                               C_Ns_dB, C_Is_dB, rho_dB)

        self.MIs_sum = np.zeros((len(C_Is_dB), len(C_Ns_dB)), dtype=np.float32)
        gMI_sum[(len(C_Is_dB), len(C_Ns_dB)),1](self.MIs_sum,
                                                self.MI0s, self.MI1_0s, self.MI1s, self.MI0_1s)

        self.Rs = np.zeros((len(C_Is_dB), len(C_Ns_dB)), dtype=np.float32)
        gMI_max_min[(len(C_Is_dB), len(C_Ns_dB)),1](self.Rs, self.MIs_sum, self.MI0s, self.MI0_1s)


    def _get_rates(self, identifier):
        '''
        identifier: 'R', 'MI0', MI1', 'MI0_1', 'MI1_0'
        '''
        if identifier == 'R':
            MIs = self.Rs
        elif identifier == 'MI0':
            MIs = self.MI0s
        elif identifier == 'MI1':
            MIs = self.MI1s
        elif identifier == 'MI0_1':
            MIs = self.MI0_1s
        elif identifier == 'MI1_0':
            MIs = self.MI1_0s
        return MIs
    
    def imshow(self, ax, identifier):
        MIs = self._get_rates(identifier)
        extent = [self.C_Ns_dB[0], self.C_Ns_dB[-1], self.C_Is_dB[-1], self.C_Is_dB[0]]
        h = ax.imshow(MIs, extent=extent)
        ax.set_xlabel('C/N in dB')
        ax.set_ylabel('C/I in dB')
        return h

    def contour(self, rate, ax, identifier, color):
        MIs = self._get_rates(identifier)
        extent = [self.C_Ns_dB[0], self.C_Ns_dB[-1], self.C_Is_dB[-1], self.C_Is_dB[0]]
        h = ax.contour(self.C_Ns_dB, self.C_Is_dB, MIs, [rate], colors=color, linewidths=2)
        return h
