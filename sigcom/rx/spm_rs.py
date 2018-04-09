import numpy as np
from numba import njit
from sigcom.rx.util import _max_star


@njit
def demap0(rx, X0, X1, Powers0, Powers1, phase,  P_noise, N_fec_cells, La0, La1):
    N_cells = len(rx)
    M0 = len(X0)
    M1 = len(X1)
    ldM0 = int(np.log2(M0))
    ldM1 = int(np.log2(M1))
    N_bits = N_cells*ldM0

    Cp_sqrt  = np.sqrt(Powers0[0])
    Cpp_sqrt = np.sqrt(Powers0[1]) 
    Ip_sqrt  = np.sqrt(Powers1[0])
    Ipp_sqrt = np.sqrt(Powers1[1]) 

    Llrs = np.zeros(N_bits)
    for k in range(N_cells):
        if int(k/(N_fec_cells/2)) % 2 == 0:
            C_sqrt = Cp_sqrt
            I_sqrt = Ip_sqrt
        else:
            C_sqrt = Cpp_sqrt
            I_sqrt = Ipp_sqrt
        D = np.zeros(M0*M1)
        for i in range(M0):
            for j in range(M1):
                d = rx[k] - C_sqrt*X0[i] - I_sqrt*X1[j]*phase[k]
                D[j*M0+i] = -1/P_noise*np.abs(d)**2
                if len(La0) > 0:
                    for m in range(ldM0):
                        bit = (i >> (ldM0 - 1 - m)) & 1
                        D[j*M0+i] -= bit*La0[k*ldM0+m]
                if len(La1) > 0:
                    for m in range(ldM1):
                        bit = (j >> (ldM1 - 1 - m)) & 1
                        D[j*M0+i] -= bit*La1[k*ldM1+m]
        for m in range(ldM0):
            num = den = np.float64(-np.inf)
            for i in range(M0):
                for j in range(M1):
                    if (i >> (ldM0 - 1 - m)) & 1:
                        den = _max_star(den, D[j*M0+i])
                    else:
                        num = _max_star(num, D[j*M0+i])
            Llrs[k*ldM0+m] = num - den
    return Llrs

@njit
def demap1(rx, X0, X1, Powers0, Powers1, phase, P_noise, N_fec_cells, La0, La1):
    N_cells = len(rx)
    M0 = len(X0)
    M1 = len(X1)
    ldM0 = int(np.log2(M0))
    ldM1 = int(np.log2(M1))
    N_bits = N_cells*ldM1

    Cp_sqrt  = np.sqrt(Powers0[0])
    Cpp_sqrt = np.sqrt(Powers0[1]) 
    Ip_sqrt  = np.sqrt(Powers1[0])
    Ipp_sqrt = np.sqrt(Powers1[1]) 

    Llrs = np.zeros(N_bits)
    for k in range(N_cells):
        if int(k/(N_fec_cells/2)) % 2 == 0:
            C_sqrt = Cp_sqrt
            I_sqrt = Ip_sqrt
        else:
            C_sqrt = Cpp_sqrt
            I_sqrt = Ipp_sqrt
        D = np.zeros(M0*M1)
        for i in range(M0):
            for j in range(M1):
                d = rx[k] - C_sqrt*X0[i] - I_sqrt*X1[j]*phase[k]
                D[j*M0+i] = -1/P_noise*np.abs(d)**2
                if len(La0) > 0:
                    for m in range(ldM0):
                        bit = (i >> (ldM0 - 1 - m)) & 1
                        D[j*M0+i] -= bit*La0[k*ldM0+m]
                if len(La1) > 0:
                    for m in range(ldM1):
                        bit = (j >> (ldM1 - 1 - m)) & 1
                        D[j*M0+i] -= bit*La1[k*ldM1+m]
        for m in range(ldM1):
            num = den = np.float64(-np.inf)
            for i in range(M0):
                for j in range(M1):
                    if (j >> (ldM1 - 1 - m)) & 1:
                        den = _max_star(den, D[j*M0+i])
                    else:
                        num = _max_star(num, D[j*M0+i])
            Llrs[k*ldM1+m] = num - den
    return Llrs
