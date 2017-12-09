from scipy.sparse import csc_matrix, csr_matrix
import numpy as np
from collections import namedtuple


class CodeParam():
    def __init__(self, pck, N, code_rate_id, nCyclicFactor):
        self.N = N
        self.pck = np.array([np.array(x, dtype=np.int32) for x in pck])
        self.CR = code_rate_id[0] / code_rate_id[1]
        self.K = int(N * self.CR)
        self.code_rate_id = code_rate_id
        self.nCyclicFactor = nCyclicFactor


def _pck_to_sparse_rows_and_cols(cp):
    '''
    cp is short for CodeParam

    Take the definition of the parity check matrices from DVB or ATSC
    and translate them to a list representation of all edges.
    Call 'H = csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(N-K,N))'
    afterwards to get a sparse matrix representation.
    '''
    nLayers = int((cp.N-cp.K)/cp.nCyclicFactor)
    rows = []
    cols = []
    col = 0
    for row in cp.pck:
        for m in range(cp.nCyclicFactor):
            for p in row:
                idx = (p+m*nLayers) % (cp.N-cp.K)
                rows.append(idx)
                cols.append(col)
            col += 1
    for k in range(cp.N-cp.K-1):
        rows.extend([k, k+1])
        cols.extend([col, col])
        col += 1
    rows.append(cp.N-cp.K-1)
    cols.append(col)
    return rows, cols


def make_pck(cp):
    rows, cols = _pck_to_sparse_rows_and_cols(cp)
    shape = (cp.N-cp.K, cp.N)
    H = csc_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
    return H


def get_parity_interleaver(K, N=64800, nCyclicFactor=360):
    parity = np.arange(N-K).reshape((nCyclicFactor, -1)).T.reshape(-1)
    return np.hstack((np.arange(K), parity+K))


def get_layerwise_pck(cp, isParityPermuted):
    '''
    layerwise_pcks, diagOffsets = get_layerwise_pck(code, isParityPermuted)
    '''
    layerwise_pcks = []
    N_layers = int((cp.N-cp.K)/cp.nCyclicFactor)
    diagOffsets = np.zeros(N_layers)
    for layer in range(N_layers):
        pck = []
        for segment, ParityChecks in enumerate(cp.pck):
            CurrentLayers = (ParityChecks - layer) % N_layers
            VerticalShifts = (ParityChecks[CurrentLayers == 0] - layer) / N_layers
            if len(VerticalShifts) > 0:
                DiagonalShifts = (cp.nCyclicFactor-VerticalShifts) % cp.nCyclicFactor
                ColumnOffset = segment*cp.nCyclicFactor
                pck.extend([[ColumnOffset, int(ds)] for ds in DiagonalShifts])

        if isParityPermuted:
            pck.append([cp.K+cp.nCyclicFactor*layer, 0])
            if layer == 0:
                pck.append([cp.N-cp.nCyclicFactor, cp.nCyclicFactor-1])
            else:
                pck.append([cp.K+cp.nCyclicFactor*(layer-1), 0])
        else:
            pck.append([cp.K, layer])
            pck.append([cp.K, layer-1])
        layerwise_pcks.append(pck)

        if layer+1 < N_layers:
            diagOffsets[layer+1] = int(diagOffsets[layer] + len(pck))

    layerwise_pcks = np.array([np.array(x, dtype=np.int32) for x in layerwise_pcks])
    diagOffsets = np.array(diagOffsets)
    return layerwise_pcks, diagOffsets


def layerwise_pcks_to_PCM(layerwise_pcks, cp):
    '''
    run 'layerwise_pcks, diagOffsets = get_layerwise_pck(code, isParityPermuted)'
    '''
    cols = []
    rows = []
    row = 0
    for i, pck in enumerate(layerwise_pcks):
        for m in range(cp.nCyclicFactor):
            x = pck[:, 0] + (pck[:, 1] + m) % cp.nCyclicFactor
            if (i == 0) and (m == 0):
                x = x[:-1]
            cols.extend(x)
            rows.extend([row]*len(x))
            row += 1

    shape = (cp.N-cp.K, cp.N)
    PCM = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)
    return PCM


if __name__ == '__main__':
    from sigcom.coding.atsc import pck_long
    codeParam = pck_long.get_pck([8, 15])
    H = make_pck(codeParam)
    isParityPermuted = True
    layerwise_pcks, diagOffsets = get_layerwise_pck(codeParam, isParityPermuted)
    PCM = layerwise_pcks_to_PCM(layerwise_pcks, codeParam)
    import matplotlib.pyplot as plt
    plt.spy(PCM, markersize=.1)
    plt.show()
