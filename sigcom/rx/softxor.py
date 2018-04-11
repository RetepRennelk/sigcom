import numpy as np
from numba import njit
import numba as nb


@njit
def _correctionTerm(x):
    return np.log(1.0+np.exp(-x))


@njit
def _partialSoftXor(L1, L2):
    L1_abs = np.abs(L1)
    L2_abs = np.abs(L2)
    rhs0 = np.min(np.array([L1_abs, L2_abs]))
    rhs1 = _correctionTerm(L1_abs+L2_abs)
    rhs2 = _correctionTerm(np.abs(L1_abs-L2_abs))
    return np.sign(L1)*np.sign(L2)*(rhs0+rhs1-rhs2)


@njit
def softXor(Llrs):
    out = Llrs[0]
    for Llr in Llrs[1:]:
        out = _partialSoftXor(out, Llr)
    return out


@njit
def softXorFwdBwd(Llrs):
    N = len(Llrs)

    ExtrinsicBwd = np.zeros(N)
    ExtrinsicBwd[-1] = np.inf
    for k in range(N-1, 0, -1):
        ExtrinsicBwd[k-1] = _partialSoftXor(ExtrinsicBwd[k], Llrs[k])

    ExtrinsicFwd = np.zeros(N)
    ExtrinsicFwd[0] = np.inf
    for k in range(N-1):
        ExtrinsicFwd[k+1] = _partialSoftXor(ExtrinsicFwd[k], Llrs[k])

    Extrinsic = np.zeros(N)
    for k in range(N):
        Extrinsic[k] = _partialSoftXor(ExtrinsicFwd[k], ExtrinsicBwd[k])
    return Extrinsic


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

    print(softXorFwdBwd(Llrs))
