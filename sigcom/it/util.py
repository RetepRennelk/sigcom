import numpy as np


def getNoisePower(Ias):
    '''
    See bits_to_apriori(bits, Ia) how getNoisePower is put to use
    '''
    Pas = []
    for Ia in np.atleast_1d(Ias):
        if Ia <= .3646:
            a1 = 1.09542
            b1 = 0.214217
            c1 = 2.33727
            Pa = (a1*Ia**2+b1*Ia+c1*np.sqrt(Ia))**2
        else:
            a2 = 0.706692
            b2 = 0.386013
            c2 = -1.75017
            Pa = (-a2*np.log(b2*(1-Ia))-c2*Ia)**2
        Pas.append(Pa)
    return np.array(Pas) if len(Pas) > 1 else Pas[0]


def getMutualInfo(Pas):
    Ias = []
    for Pa in np.atleast_1d(Pas):
        nSigma = np.sqrt(Pa)
        if nSigma >= 10:
            Ia = 1
        elif nSigma <= 1.6363:
            a1 = -0.0421061
            b1 = 0.209252
            c1 = -.00640081
            Ia = a1*nSigma**3+b1*nSigma**2+c1*nSigma
        else:
            a2 = 0.00181491
            b2 = -0.142675
            c2 = -0.0822054
            d2 = 0.0549608
            Ia = 1-np.exp(a2*nSigma**3+b2*nSigma**2+c2*nSigma+d2)
        Ias.append(Ia)
    return np.array(Ias) if len(Ias) > 1 else Ias[0]


def mutual_information_magic(Llrs, bits, ldM):
    MIs = []
    for m in range(ldM):
        b = bits[m::ldM]
        L = Llrs[m::ldM]
        MI = 1-np.mean(np.log2(1+np.exp(-(1-2*b)*L)))
        MIs.append(MI)
    return np.array(MIs) if ldM > 1 else MIs[0]


def bits_to_apriori(bits, Ia):
    noise = np.random.randn(len(bits))
    Pa = getNoisePower(Ia)
    Llrs = Pa/2*(1-2*bits) + np.sqrt(Pa)*noise
    return Llrs


if __name__ == '__main__':
    from sigcom.tx.util import generate_bits
    N_bits = 100000
    bits = generate_bits(N_bits)
    Ia = .6
    Llrs = bits_to_apriori(bits, Ia)
    print(mutual_information_magic(Llrs, bits, ldM=1))
