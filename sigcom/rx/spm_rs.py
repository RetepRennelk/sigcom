import numpy as np
from numba import njit
from sigcom.rx.util import _max_star


@njit
def demap0(rx, X0, X1, Ps, h0, h1, P_noise, N_fec_cells, La0, La1):
    N_cells = len(rx)
    M0 = len(X0)
    M1 = len(X1)
    ldM0 = int(np.log2(M0))
    ldM1 = int(np.log2(M1))
    N_bits = N_cells*ldM0

    P0_sqrt = np.sqrt(Ps[0])
    P1_sqrt = np.sqrt(Ps[1]) 

    Llrs = np.zeros(N_bits)
    for k in range(N_cells):
        sw = int(k/(N_fec_cells/2)) % 2
        if sw == 0:
            S0 = P0_sqrt
            S1 = P1_sqrt
        else:
            S0 = P1_sqrt
            S1 = P0_sqrt
        D = np.zeros(M0*M1)
        for i in range(M0):
            for j in range(M1):
                d = rx[k] - S0*h0[k]*X0[i] - S1*h1[k]*X1[j]
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
def demap1(rx, X0, X1, Ps, h0, h1, P_noise, La0, La1):
    N_cells = len(rx)
    M0 = len(X0)
    M1 = len(X1)
    ldM0 = int(np.log2(M0))
    ldM1 = int(np.log2(M1))
    N_bits = N_cells*ldM1

    Llrs = np.zeros(N_bits)
    for k in range(N_cells):
        D = np.zeros(M0*M1)
        for i in range(M0):
            for j in range(M1):
                d = rx[k] - np.sqrt(Ps[0])*h0[k]*X0[i]
                d-= np.sqrt(Ps[1])*h1[k]*X1[j]
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
