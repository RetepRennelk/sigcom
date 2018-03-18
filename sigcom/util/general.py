import numpy as np
import functools


def ints_to_bits(ints, bw):
    '''
    Convert integers 'ints' to bits with bitwidth 'bw'

    Parameters:
    - ints: list of integers
    - bw:  bitwidth to which the integers are converted
    '''
    l = [((i >> bw-1-j)) & 1 for i in ints for j in range(bw)]
    return np.array(l, dtype=np.int)


def bits_to_ints(bits, bw):
    '''
    Convert bits to integers with bitwidth 'bw'
    '''
    weights = 2**np.arange(bw)[::-1]
    ints = bits.reshape(-1, bw).dot(weights)
    return np.array(ints, np.int)


@functools.lru_cache(maxsize=None)
def hermite_poly(n):
    '''
    H0(x) = 1
    H1(x) = 2x
    H_(n)(x) = 2*x*H_(n-1)(x) - 2*(n-1)*H_(n-2)(x)
    '''
    if n == 0:
        return np.array([1.])
    elif n == 1:
        return np.array([2., 0.])
    else:
        out = np.append(hermite_poly(n-1), 0.)
        out[2:] -= (n-1)*hermite_poly(n-2)
        return 2*out


def gauss_hermite_weights_abscissas(n=32):
    '''
    wi, xi = gauss_hermite_weights_abscissas(n=32)
    '''
    H_n = hermite_poly(n)
    xi = np.roots(H_n)

    H_n_1 = hermite_poly(n-1)
    num = 2**(n-1) * np.math.factorial(n-1) * np.sqrt(np.pi)
    den = n * np.polyval(H_n_1, xi)**2
    wi = num / den
    return wi, xi


class MI_ten_brink():
    def __int__(self, N):
        self.N = N

    @staticmethod
    def getMutualInfo(P):
        def helper(P):
            sigma = np.sqrt(P)
            if sigma >= 10:
                Ia = 1.
            elif sigma <= 1.6363:
                a = -.0421061
                b = .209252
                c = -.00640081
                Ia = a*sigma**3 + b*sigma**2 + c*sigma
            else:
                a = .00181491
                b = -.142675
                c = -.0822054
                d = .0549608
                Ia = 1-np.exp(a*sigma**3 + b*sigma**2 + c*sigma + d)
            return Ia
        return np.array([helper(p) for p in np.atleast_1d(P)])

    @staticmethod
    def getNoisePower(Ia):
        def helper(Ia):
            if Ia <= .3646:
                a = 1.09542
                b = .214217
                c = 2.33727
                Pa = (a*Ia**2 + b*Ia + c*np.sqrt(Ia))**2
            else:
                a = .706692
                b = .386013
                c = -1.75017
                Pa = (-a*np.log(b*(1-Ia)) - c*Ia)**2
            return Pa
        return np.array([helper(ia) for ia in np.atleast_1d(Ia)])

    def update(self):
        self.noise = np.random.randn(self.N)

    def add_apriori_info(self, bits, Ia):
        Pa = MI_ten_brink.getNoisePower(Ia)
        return Pa/2*(1-2*bits) + np.sqrt(Pa)*self.noise


class MI_Gauss_Hermite():
    '''
    Mutual Information for a Binary AWGN Channel
    '''
    def __init__(self, n=32):
        self.n = n
        wi, xi = gauss_hermite_weights_abscissas(n)

        def func_helper(t, x, P):
            return 1/np.sqrt(np.pi)*np.log2(1+np.exp(-2*x*(np.sqrt(2*P)*t+x)/P))
        self.func = lambda P: 1-.5*(func_helper(xi, +1, P)+func_helper(xi, -1, P)).dot(wi)

    def get(self, P):
        return np.array([self.func(4/p) for p in np.atleast_1d(P)])

    def plot(self):
        import matplotlib.pyplot as plt
        P_dB = np.linspace(-40, 40, 100)
        plt.plot(P_dB, self.get(10**(P_dB/10)))
        plt.show()


class Struct():
    '''
    Convenience class to collect a bunch of named data.

    Example:
    struct = Struct(a=12, b=4)

    print(struct.a)
    print(struct.b)
    '''
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


if __name__ == '__main__':
    ints = range(5)
    bits = ints_to_bits(ints, 3)
    print(bits.reshape(-1, 3))
