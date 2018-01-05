'''
Utilities for Galois Field computations
'''

import numpy as np

# Dictionary of Primitive Polynomials
# The lists contain the non-zero exponents of the PPs
pp_exp = {2: [0, 1, 2], 3: [0, 1, 3], 4: [0, 1, 4], 5: [0, 2, 5], 6: [0, 1, 6],
          7: [0, 3, 7], 8: [0, 2, 3, 4, 8], 9: [0, 4, 9], 10: [0, 3, 10],
          11: [0, 2, 11], 12: [0, 1, 4, 6, 12], 13: [0, 1, 3, 4, 13],
          14: [0, 1, 6, 10, 14], 15: [0, 1, 15], 16: [0, 1, 3, 12, 16],
          17: [0, 3, 17], 18: [0, 7, 18], 19: [0, 1, 2, 5, 19], 20: [0, 3, 20],
          21: [0, 2, 21], 22: [0, 1, 22], 23: [0, 5, 23], 24: [0, 1, 2, 7, 24]}


class GF():
    def __init__(self, m):
        self.m = m
        self.pp_exp = pp_exp[m]
        self.int_rep = self._pp_exp_to_int()

    def _pp_exp_to_int(self):
        degree = self.pp_exp[-1]
        polynomial = np.zeros(degree+1)
        polynomial[self.pp_exp] = 1
        weights = 2**np.arange(degree+1)
        return np.int(polynomial.dot(weights))

    def add(self, a, b):
        return a ^ b

    def mul(self, a, b):
        p = 0  # the product of the multiplication
        while a and b:
            # if b is odd, then add the corresponding a to p
            # final product = sum of all a's corresponding to odd b's
            if b & 1:
                p ^= a  # addition equals XOR in GF(2)
            # avoid overfloww
            if a >> self.m - 1:
                # XOR with the primitive polynomial
                a = (a << 1) ^ self.int_rep
            else:
                a <<= 1
            b >>= 1
        return p


if __name__ == '__main__':
    m = 3
    gf = GF(m)
    print(gf.int_rep)
    for a in range(2**m):
        print([gf.mul(a,b) for b in range(2**m)])
