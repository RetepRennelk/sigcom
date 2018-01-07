'''
Utilities for Galois Field computations
'''

from copy import deepcopy
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

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

    def cumprod(self, arr):
        p = arr[0]
        for el in arr[1, :]:
            p = self.mul(p, el)
        return p


def bits_to_gf_symbols(bits, m):
    weights = 2**np.arange(m)[::-1]
    return bits.reshape(-1, m).dot(weights)


def pcm_layerwise(layerwise_pcks, N_rows, N_cols):
    rows = []
    cols = []
    for row, pcks in enumerate(layerwise_pcks):
        rows.extend([row]*len(pcks))
        cols.extend(pcks)
    shape = (N_rows, N_cols)
    return csr_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=shape, dtype=np.int)


def pcm_columnwise(columnwise_pcks, N_rows, N_cols):
    cols = []
    rows = []
    for col, pcks in enumerate(columnwise_pcks):
        cols.extend([col]*len(pcks))
        rows.extend(pcks)
    shape = (N_rows, N_cols)
    return csc_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=shape, dtype=np.int)


class BasePcks():
    '''
    Interface to unify handling of layerwise and columnwise
    oriented parity check node description. Once BasePcks has
    been instantiated with either layerwise or columnwise pcks,
    both are simultaneosuly present in the instance of BasePcks.

    This class is limited to the information part of the parity
    check matrix. It explicitly excludes the parity part.

    layerwise_pcks: row by row description of the parity check addresses
    columnwise_pcks: column by column description of the parity check addresses
    '''
    def __init__(self, pcks, orientation, N, K):
        self.N = N
        self.K = K
        if orientation == 'layerwise':
            self.layerwise_pcks = deepcopy(pcks)
            self.columnwise_pcks = self.layer_to_column_pcks()
        elif orientation == 'columnwise':
            self.columnwise_pcks = deepcopy(pcks)
            self.layerwise_pcks = self.column_to_layer_pcks()
        else:
            assert False, 'Unknown orientation: ' + orientation

    def layer_to_column_pcks(self):
        col_pcks = [[] for i in range(self.K)]
        for layer, pck in enumerate(self.layerwise_pcks):
            for p in pck:
                if p < self.K:
                    col_pcks[p].append(layer)
        return col_pcks

    def column_to_layer_pcks(self):
        layerwise_pcks = [[] for i in range(self.N-self.K)]
        for col, pck in enumerate(self.columnwise_pcks):
            if col < self.K:
                for p in pck:
                    layerwise_pcks[p].append(col)
        return layerwise_pcks

    def pcm_layerwise(self):
        return pcm_layerwise(self.layerwise_pcks, self.N-self.K, self.K)

    def pcm_columnwise(self):
        return pcm_columnwise(self.columnwise_pcks, self.N-self.K, self.K)


class BasePcksLifted():
    def __init__(self, basePcks, CF):
        self.basePcks = basePcks
        self.CF = CF
        self.layerwise_pcks_lifted = self.lift(basePcks.layerwise_pcks)
        self.columnwise_pcks_lifted = self.lift(basePcks.columnwise_pcks)

    def lift(self, pcks):
        pcks_lifted = []
        for pck in pcks:
            pcks_lifted.append([[p*self.CF, 0] for p in pck])
        return pcks_lifted

    def randomize_offsets(self):
        col_pcks = self.columnwise_pcks_lifted
        col_indices = np.zeros(self.basePcks.K, dtype=np.int)
        for layer, pcks in enumerate(self.layerwise_pcks_lifted):
            for pck in pcks:
                offset = np.random.randint(self.CF)
                pck[-1] = offset
                col_idx = col_indices[pck[0]//self.CF]
                col_pcks[pck[0]//self.CF][col_idx][-1] = offset
                col_indices[pck[0]//self.CF] += 1

class Pcks():
    '''
    Add parity part to the information part of the pck description.
    '''
    def __init__(self, basePcks):
        self.basePcks = basePcks
        self.layerwise_pcks = deepcopy(basePcks.layerwise_pcks)
        self.columnwise_pcks = deepcopy(basePcks.columnwise_pcks)
        self.add_parity_to_layerwise_pcks()
        self.add_parity_to_columnwise_pcks()

    def add_parity_to_layerwise_pcks(self):
        pcks = self.layerwise_pcks
        pcks[0].extend([self.basePcks.K])
        for layer in range(1, self.basePcks.N-self.basePcks.K):
            x = [self.basePcks.K+layer, self.basePcks.K+layer-1]
            pcks[layer].extend(x)

    def add_parity_to_columnwise_pcks(self):
        pcks = self.columnwise_pcks
        for layer in range(self.basePcks.N-self.basePcks.K-1):
            x = [layer, layer+1]
            pcks.append(x)
        pcks.append([self.basePcks.N-self.basePcks.K-1])

    def pcm_layerwise(self):
        N_rows = self.basePcks.N-self.basePcks.K
        N_cols = self.basePcks.N
        return pcm_layerwise(self.layerwise_pcks, N_rows, N_cols)

    def pcm_columnwise(self):
        N_rows = self.basePcks.N-self.basePcks.K
        N_cols = self.basePcks.N
        return pcm_columnwise(self.columnwise_pcks, N_rows, N_cols)


class CodeParamUnlifted():
    def __init__(self, base_pck, N, code_rate_id):
        self.N = N
        self.CR = code_rate_id[0] / code_rate_id[1]
        self.K = int(N * self.CR)
        self.code_rate_id = code_rate_id
        self.base_pck = base_pck
        par_pck = self.add_parity_to_base_pck()
        self.par_pck = np.array([np.array(x, dtype=np.int32) for x in par_pck])

    def add_parity_to_base_pck(self):
        par_pck = deepcopy(self.base_pck)
        par_pck[0].extend([self.K])
        for layer in range(1, self.N-self.K):
            x = [self.K+layer, self.K+layer-1]
            par_pck[layer].extend(x)
        return par_pck

    def par_pcm(self):
        '''
        Returns the parity check matrix with parity part
        in Compressed Sparse Column (csc) format.
        '''
        rows = []
        cols = []
        for row, pck in enumerate(self.par_pck):
            cols.extend(pck)
            rows.extend([row]*len(pck))
        shape = (self.N - self.K, self.N)
        return csc_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)

    def base_pcm(self):
        '''
        Returns the base parity check matrix without parity part
        in Compressed Sparse Column (csc) format.
        '''
        rows = []
        cols = []
        for row, pck in enumerate(self.base_pck):
            cols.extend(pck)
            rows.extend([row]*len(pck))
        shape = (self.N - self.K, self.K)
        return csc_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)

    @staticmethod
    def test_instance():
        base_pck = [[0, 2], [0, 3], [1, 2], [1, 3]]
        N = 8
        code_rate_id = [1, 2]
        return CodeParamUnlifted(base_pck, N, code_rate_id)


if __name__ == '__main__':
    m = 3
    gf = GF(m)
    print(gf.int_rep)
    for a in range(2**m):
        print([gf.mul(a,b) for b in range(2**m)])
