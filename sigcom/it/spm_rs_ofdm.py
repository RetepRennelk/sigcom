import numpy as np
import matplotlib.pyplot as plt
from sigcom.tx.modcod import ModCodSP1p4_rs, ModCodSP_rs_ricean
from sigcom.ch.util import make_noise
from sigcom.rx.spm_rs import demap0, demap1
from numba import njit
from sigcom.rx.util import _max_star


@njit
def _MI0(rx, tx0, Powers0, Powers1, X0, X1, h0, h1, P_noise, N_fec_cells):
    N_cells = len(rx)
    N_codewords = N_cells // N_fec_cells
    m_rx = rx.reshape(-1, N_fec_cells)
    m_tx0 = tx0.reshape(-1, N_fec_cells)
    m_h0 = h0.reshape(-1, N_fec_cells)
    m_h1 = h1.reshape(-1, N_fec_cells)

    Cp_sqrt  = np.sqrt(Powers0[0])
    Cpp_sqrt = np.sqrt(Powers0[1]) 
    Ip_sqrt  = np.sqrt(Powers1[0])
    Ipp_sqrt = np.sqrt(Powers1[1]) 
    
    MI0s = []
    for i in range(N_codewords):
        MI0 = 0
        for k in range(N_fec_cells):
            if 2*k//N_fec_cells % 2 == 0:
                C_sqrt = Cp_sqrt
                I_sqrt = Ip_sqrt
            else:
                C_sqrt = Cpp_sqrt
                I_sqrt = Ipp_sqrt

            num = -np.inf
            den = -np.inf
            for x1 in X1:
                D = m_rx[i,k] - C_sqrt*m_h0[i,k]*m_tx0[i,k] - I_sqrt*x1*m_h1[i,k]
                num = _max_star(num, -np.abs(D)**2/P_noise)
                for x0 in X0:
                    D = m_rx[i,k] - C_sqrt*x0*m_h0[i,k] - I_sqrt*x1*m_h1[i,k]
                    den = _max_star(den, -np.abs(D)**2/P_noise)
            MI0 += num - den
        MI0s.append(np.log2(len(X0)) + MI0/N_fec_cells/np.log(2.0))
    return np.array(MI0s)

@njit
def _MI1(rx, tx1, Powers0, Powers1, X0, X1, h0, h1, P_noise, N_fec_cells):
    N_cells = len(rx)
    N_codewords = N_cells // N_fec_cells
    m_rx = rx.reshape(-1, N_fec_cells)
    m_tx1 = tx1.reshape(-1, N_fec_cells)
    m_h0 = h0.reshape(-1, N_fec_cells)
    m_h1 = h1.reshape(-1, N_fec_cells)

    Cp_sqrt  = np.sqrt(Powers0[0])
    Cpp_sqrt = np.sqrt(Powers0[1]) 
    Ip_sqrt  = np.sqrt(Powers1[0])
    Ipp_sqrt = np.sqrt(Powers1[1]) 
    
    MI1s = []
    for i in range(N_codewords):
        MI1 = 0
        for k in range(N_fec_cells):
            if 2*k//N_fec_cells % 2 == 0:
                C_sqrt = Cp_sqrt
                I_sqrt = Ip_sqrt
            else:
                C_sqrt = Cpp_sqrt
                I_sqrt = Ipp_sqrt
                
            num = -np.inf
            den = -np.inf
            for x0 in X0:
                D = m_rx[i,k] - C_sqrt*x0*m_h0[i,k] - I_sqrt*m_tx1[i,k]*m_h1[i,k]
                num = _max_star(num, -np.abs(D)**2/P_noise)
                for x1 in X1:
                    D = m_rx[i,k] - C_sqrt*x0*m_h0[i,k] - I_sqrt*x1*m_h1[i,k]
                    den = _max_star(den, -np.abs(D)**2/P_noise)
            MI1 += num - den
        MI1s.append(np.log2(len(X1))+MI1/N_fec_cells/np.log(2.0))
    return np.array(MI1s)


@njit
def _MI1_0(rx, tx0, tx1, Powers0, Powers1, X0, X1, h0, h1, P_noise, N_fec_cells):
    N_cells = len(rx)
    N_codewords = N_cells // N_fec_cells
    m_rx = rx.reshape(-1, N_fec_cells)
    m_tx0 = tx0.reshape(-1, N_fec_cells)
    m_tx1 = tx1.reshape(-1, N_fec_cells)
    m_h0 = h0.reshape(-1, N_fec_cells)
    m_h1 = h1.reshape(-1, N_fec_cells)
    
    Cp_sqrt  = np.sqrt(Powers0[0])
    Cpp_sqrt = np.sqrt(Powers0[1]) 
    Ip_sqrt  = np.sqrt(Powers1[0])
    Ipp_sqrt = np.sqrt(Powers1[1]) 
    
    MI1_0s = []
    for i in range(N_codewords):
        MI1 = 0
        for k in range(N_fec_cells):
            if 2*k//N_fec_cells % 2 == 0:
                C_sqrt = Cp_sqrt
                I_sqrt = Ip_sqrt
            else:
                C_sqrt = Cpp_sqrt
                I_sqrt = Ipp_sqrt
        
            D = m_rx[i,k] - C_sqrt*m_tx0[i,k]*m_h0[i,k] - I_sqrt*m_tx1[i,k]*m_h1[i,k]
            num = -np.abs(D)**2/P_noise
            
            den = -np.inf
            for x1 in X1:
                D = m_rx[i,k] - C_sqrt*m_tx0[i,k]*m_h0[i,k] - I_sqrt*x1*m_h1[i,k]
                den = _max_star(den, -np.abs(D)**2/P_noise)
            MI1 += num - den
        MI1_0s.append(np.log2(len(X1)) + MI1/N_fec_cells/np.log(2.0))
    return np.array(MI1_0s)

@njit
def _MI0_1(rx, tx0, tx1, Powers0, Powers1, X0, X1, h0, h1, P_noise, N_fec_cells):
    N_cells = len(rx)
    N_codewords = N_cells // N_fec_cells
    m_rx = rx.reshape(-1, N_fec_cells)
    m_tx0 = tx0.reshape(-1, N_fec_cells)
    m_tx1 = tx1.reshape(-1, N_fec_cells)
    m_h0 = h0.reshape(-1, N_fec_cells)
    m_h1 = h1.reshape(-1, N_fec_cells)
    
    Cp_sqrt  = np.sqrt(Powers0[0])
    Cpp_sqrt = np.sqrt(Powers0[1]) 
    Ip_sqrt  = np.sqrt(Powers1[0])
    Ipp_sqrt = np.sqrt(Powers1[1]) 
    
    MI0_1s = []
    for i in range(N_codewords):
        MI0_1 = 0
        for k in range(N_fec_cells):
            if 2*k//N_fec_cells % 2 == 0:
                C_sqrt = Cp_sqrt
                I_sqrt = Ip_sqrt
            else:
                C_sqrt = Cpp_sqrt
                I_sqrt = Ipp_sqrt
                
            D = m_rx[i,k] - C_sqrt*m_tx0[i,k]*m_h0[i,k] - I_sqrt*m_tx1[i,k]*m_h1[i,k]
            num = -np.abs(D)**2/P_noise
            
            den = -np.inf
            for x0 in X0:
                D = m_rx[i,k] - C_sqrt*x0*m_h0[i,k] - I_sqrt*m_tx1[i,k]*m_h1[i,k]
                den = _max_star(den, -np.abs(D)**2/P_noise)
            MI0_1 += num - den
        MI0_1s.append(np.log2(len(X0))+MI0_1/N_fec_cells/np.log(2.0))
    return np.array(MI0_1s)

class MI_SP1p4_rs_ofdm():
    def __init__(self, M, CR, N_ldpc):
        self.mc = ModCodSP1p4_rs(M, CR, N_ldpc)
        self.N_fec_cells = int(N_ldpc / np.log2(M))

    def update(self, N_codewords):
        self.N_codewords = N_codewords
        self.mc.update(N_codewords)
        self.noise = make_noise(self.N_fec_cells*N_codewords)

    def generate(self, C_I_dB, C_N_dB, rho_dB):
        if 1:
            N = 1
            C_I = 10**(C_I_dB/10)
            C_N = 10**(C_N_dB/10)
            rho = 10**(rho_dB/10)
            C = C_N*N
            self.P_noise = N
            self.Powers0 = [2*rho/(1+rho)*C, 2/(1+rho)*C]
            self.Powers1 = [2/(1+rho)*C/C_I, 2*rho/(1+rho)*C/C_I]
            self.mc.generate(self.Powers0, self.Powers1)
            self.rx = self.mc.tx + self.noise*np.sqrt(self.P_noise)
        elif 0:
            C = 1
            C_I = 10**(C_I_dB/10)
            C_N = 10**(C_N_dB/10)
            rho = 10**(rho_dB/10)
            self.P_noise = C/C_N
            self.Powers0 = [2*rho/(1+rho)*C, 2/(1+rho)*C]
            self.Powers1 = [2/(1+rho)*C/C_I, 2*rho/(1+rho)*C/C_I]
            self.mc.generate(self.Powers0, self.Powers1)
            self.rx = self.mc.tx + self.noise*np.sqrt(self.P_noise)

    def generate_snr(self, C_I_dB, SNR_dB, rho_dB):
        self.P_noise = 1
        C_I = 10**(C_I_dB/10)
        SNR = 10**(SNR_dB/10)
        rho = 10**(rho_dB/10)
        self.Powers0 = [rho/(1+rho)*2*SNR/(1+1/C_I), 1/(1+rho)*2*SNR/(1+1/C_I)]
        self.Powers1 = [1/(1+rho)*2*SNR/(1+C_I), rho/(1+rho)*2*SNR/(1+C_I)]
        self.mc.generate(self.Powers0, self.Powers1)
        self.rx = self.mc.tx + self.noise*np.sqrt(self.P_noise)

    def compute_MIs(self):
        N_cells = self.N_fec_cells*self.N_codewords
        self.MI0s = _MI0(self.rx, self.mc.tx0.tx, 
                         self.Powers0, self.Powers1, 
                         self.mc.tx0.X, self.mc.tx1.X, 
                         np.ones(N_cells), self.mc.phase,
                         self.P_noise, self.N_fec_cells)
        self.MI1s = _MI1(self.rx, self.mc.tx1.tx, 
                         self.Powers0, self.Powers1, 
                         self.mc.tx0.X, self.mc.tx1.X, 
                         np.ones(N_cells), self.mc.phase,
                         self.P_noise, self.N_fec_cells)
        self.MI0_1s = _MI0_1(self.rx, self.mc.tx0.tx, self.mc.tx1.tx, 
                             self.Powers0, self.Powers1, 
                             self.mc.tx0.X, self.mc.tx1.X, 
                             np.ones(N_cells), self.mc.phase,
                             self.P_noise, self.N_fec_cells)
        self.MI1_0s = _MI1_0(self.rx, self.mc.tx0.tx, self.mc.tx1.tx, 
                             self.Powers0, self.Powers1, 
                             self.mc.tx0.X, self.mc.tx1.X, 
                             np.ones(N_cells), self.mc.phase,
                             self.P_noise, self.N_fec_cells)
        self.MI_sums = 0.5*(self.MI0_1s+self.MI0s+self.MI1_0s+self.MI1s)


class MI_SP_ricean_rs_ofdm():
    def __init__(self, M, CR, N_ldpc):
        self.mc = ModCodSP_rs_ricean(M, CR, N_ldpc)
        self.N_fec_cells = int(N_ldpc / np.log2(M))

    def update(self, K_factor, N_codewords):
        self.N_codewords = N_codewords
        self.mc.update(K_factor, N_codewords)
        self.noise = make_noise(self.N_fec_cells*N_codewords)

    def generate(self, C_I_dB, C_N_dB, rho_dB):
        C = 1
        C_I = 10**(C_I_dB/10)
        C_N = 10**(C_N_dB/10)
        rho = 10**(rho_dB/10)
        self.P_noise = C/C_N
        self.Powers0 = [2*rho/(1+rho)*C, 2/(1+rho)*C]
        self.Powers1 = [2/(1+rho)*C/C_I, 2*rho/(1+rho)*C/C_I]
        self.mc.generate(self.Powers0, self.Powers1)
        self.rx = self.mc.tx + self.noise*np.sqrt(self.P_noise)

    def compute_MIs(self):
        N_cells = self.N_fec_cells*self.N_codewords
        self.MI0s = _MI0(self.rx, self.mc.tx0.tx, 
                         self.Powers0, self.Powers1, 
                         self.mc.tx0.X, self.mc.tx1.X,
                         self.mc.h0, self.mc.h1,
                         self.P_noise, self.N_fec_cells)
        self.MI1s = _MI1(self.rx, self.mc.tx1.tx, 
                         self.Powers0, self.Powers1, 
                         self.mc.tx0.X, self.mc.tx1.X, 
                         self.mc.h0, self.mc.h1,
                         self.P_noise, self.N_fec_cells)
        self.MI0_1s = _MI0_1(self.rx, self.mc.tx0.tx, self.mc.tx1.tx, 
                             self.Powers0, self.Powers1, 
                             self.mc.tx0.X, self.mc.tx1.X, 
                             self.mc.h0, self.mc.h1,
                             self.P_noise, self.N_fec_cells)
        self.MI1_0s = _MI1_0(self.rx, self.mc.tx0.tx, self.mc.tx1.tx, 
                             self.Powers0, self.Powers1, 
                             self.mc.tx0.X, self.mc.tx1.X, 
                             self.mc.h0, self.mc.h1,
                             self.P_noise, self.N_fec_cells)
        self.MI_sums = 0.5*(self.MI0_1s+self.MI0s+self.MI1_0s+self.MI1s)
