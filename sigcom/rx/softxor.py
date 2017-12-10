import numpy as np
from numba import njit


def softXor(Llrs):
    @njit
    def ct(x):
        # Correction Term
        return np.log(1.0+np.exp(-x))

    @njit
    def partial(L1, L2):
        L1_abs = np.abs(L1)
        L2_abs = np.abs(L2)
        rhs0 = np.min(np.array([L1_abs, L2_abs]))
        rhs1 = ct(L1_abs+L2_abs)
        rhs2 = ct(np.abs(L1_abs-L2_abs))
        return np.sign(L1)*np.sign(L2)*(rhs0+rhs1-rhs2)

    out = Llrs[0]
    for Llr in Llrs[1:]:
        out = partial(out, Llr)
    return out


if __name__ == '__main__':
    Llrs = np.abs(np.random.randn(3))
    print(Llrs)
    print(softXor(Llrs))

    p0 = 1/(1+np.exp(-Llrs[0]))
    p1 = 1-p0
    q0 = 1/(1+np.exp(-Llrs[1]))
    q1 = 1-q0
    num = p0*q0+p1*q1
    den = p0*q1+p1*q0
    print(np.log(num/den))
