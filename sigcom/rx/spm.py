import numpy as np
from numba import njit
from sigcom.rx.util import _max_star


@njit
def demap0(rx, X0, X1, Ps, h0, h1, P_noise, La0, La1):
    N_cells = len(rx)
    M0 = len(X0)
    M1 = len(X1)
    ldM0 = int(np.log2(M0))
    ldM1 = int(np.log2(M1))
    N_bits = N_cells * ldM0

    Llrs = np.zeros(N_bits)
    for k in range(N_cells):
        D = np.zeros(M0*M1)
        for i in range(M0):
            for j in range(M1):
                d = rx[k] - np.sqrt(Ps[0])*h0[k]*X0[i]
                d -= np.sqrt(Ps[1])*h1[k]*X1[j]
                D[j*M0+i] = -1/P_noise*np.abs(d)**2
                if len(La0) > 0:
                    for m in range(ldM0):
                        bit = (i >> (ldM0-1-m)) & 1
                        D[j*M0+i] -= bit * La0[k*ldM0+m]
                if len(La1) > 0:
                    for m in range(ldM1):
                        bit = (j >> (ldM1-1-m)) & 1
                        D[j*M0+i] -= bit * La1[k*ldM1+m]
        for m in range(ldM0):
            num = np.float64(-np.inf)
            den = np.float64(-np.inf)
            for i in range(M0):
                for j in range(M1):
                    if (i >> (ldM0-1-m)) & 1:
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
    N_bits = N_cells * ldM1

    Llrs = np.zeros(N_bits)
    for k in range(N_cells):
        D = np.zeros(M0*M1)
        for i in range(M0):
            for j in range(M1):
                d = rx[k] - np.sqrt(Ps[0])*X0[i]*h0[k]
                d -= np.sqrt(Ps[1])*X1[j]*h1[k]
                D[j*M0+i] = -1/P_noise*np.abs(d)**2
                if len(La0) > 0:
                    for m in range(ldM0):
                        bit = (i >> (ldM0-1-m)) & 1
                        D[j*M0+i] -= bit * La0[k*ldM0+m]
                if len(La1) > 0:
                    for m in range(ldM1):
                        bit = (j >> (ldM1-1-m)) & 1
                        D[j*M0+i] -= bit * La1[k*ldM1+m]
        for m in range(ldM1):
            num = den = np.float64(-np.inf)
            for i in range(M0):
                for j in range(M1):
                    if (j >> (ldM1-1-m)) & 1:
                        den = _max_star(den, D[j*M0+i])
                    else:
                        num = _max_star(num, D[j*M0+i])
            Llrs[k*ldM1+m] = num - den
    return Llrs


if __name__ == '__main__':
    from sigcom.tx.spm import SP1p4
    from sigcom.rx.util import make_noise
    from sigcom.it.util import bits_to_apriori

    N_cells = 10000
    sp1p4 = SP1p4(N_cells)
    Ps = [1, 1]
    sp1p4.generate(Ps)
    noise = make_noise(N_cells)
    P_noise = .1
    rx = sp1p4.tx + noise*np.sqrt(P_noise)

    Ia0 = .1
    Ia1 = .9
    La0 = bits_to_apriori(sp1p4.bits0, [Ia0])
    La1 = bits_to_apriori(sp1p4.bits1, [Ia1])

    Llrs0 = demap0(rx, sp1p4.X0, sp1p4.X1, Ps, sp1p4.phase, P_noise, La0, La1)
    Llrs1 = demap1(rx, sp1p4.X0, sp1p4.X1, Ps, sp1p4.phase, P_noise, La0, La1)

    print(np.sum((Llrs0<0)!=sp1p4.bits0))
    print(np.sum((Llrs1<0)!=sp1p4.bits1))

