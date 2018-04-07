from scipy.sparse import csc_matrix, csr_matrix
import numpy as np


class CodeParam():
    def __init__(self, pck, N, code_rate_id, nCyclicFactor):
        '''
        Parameters
        - pck: Parity check edges as defined in DVB-T2 or ATSC
               in a column oriented fashion
        '''
        self.N = N
        self.pck = np.array([np.array(x, dtype=np.int32) for x in pck])
        self.CR = code_rate_id[0] / code_rate_id[1]
        self.K = int(N * self.CR)
        self.code_rate_id = code_rate_id
        self.nCyclicFactor = nCyclicFactor
        self.layerwise_pcks, _  = get_layerwise_pck(self, True)
        self.N_diags = get_num_diags(self.layerwise_pcks)


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
    layerwise_pcks, diagOffsets = get_layerwise_pck(cp, isParityPermuted)
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


def layerwise_pcks_to_string(layerwise_pcks):
    '''
    Translate layerwise_pcks to string representation.
    This is useful for CUDA-jitting.

    Example:
    [[[0, 1], [4, 3]], [[8, 2], [12, 1]]]
    ->
    '[[[0,1],[4,3]],[[8,2],[12,1]]]'
    '''
    s = '['
    for pcks in layerwise_pcks:
        s += '['
        for pck in pcks:
            s += '[{},{}],'.format(pck[0], pck[1])
        s = s[:-1] + '],'
    s_layerwise_pcks = s[:-1] + ']'
    return s_layerwise_pcks


def layerwise_pcks_to_PCM(layerwise_pcks, cp):
    '''
    run 'layerwise_pcks, diagOffsets = get_layerwise_pck(cp, isParityPermuted)'
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


def get_num_diags(layerwise_pcks):
    num_diags = 0
    for pck in layerwise_pcks:
        num_diags += len(pck)
    return num_diags


class LinLayerwisePcks():
    '''
    Put all pcks into a single list 'llpcks'.
    Keep track of the original pcks with a second
    list 'addr'

    Example:
    layerwise_pcks = [[[0,1],[2,3]],[[4,5],[6,7],[8,9]]]
    llpcks = [0,1,2,3,4,5,6,7,8,9]
    addr = [0,4,10]
    '''
    def __init__(self, layerwise_pcks):
        addr = [0]
        linearized_layerwise_pcks = []
        for pcks in layerwise_pcks:
            N_pcks = 0
            for pck in pcks:
                linearized_layerwise_pcks.extend(pck)
                N_pcks += 2
            addr.append(addr[-1]+N_pcks)
        self.addr = addr
        self.llpcks = linearized_layerwise_pcks


if __name__ == '__main__':
    if 1:
        from sigcom.coding.atsc.code_param_short import get
        from sigcom.coding.PCM import PCM
        from scipy.sparse import vstack, csr_matrix

        cp = get([8, 15])
        pcm = PCM(cp)
        bil = get_parity_interleaver(cp.K, cp.N)

        isParityPermuted = True
        H = pcm.make_layered(isParityPermuted)

        rows, cols = _pck_to_sparse_rows_and_cols(cp)
        H0 = csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(cp.N-cp.K,cp.N))
        H0_bil = H0[:, bil]
        N_layers = int((cp.N-cp.K)/360)
        H1 = H0_bil[0::N_layers, :]
        for l in range(1, N_layers):
            H1 = vstack((H1, H0_bil[l::N_layers, :]))
        H1 = csr_matrix(H1)
        for r in range(cp.N-cp.K):
            a = np.nonzero(H[r,:])[1]
            b = np.nonzero(H1[r,:])[1]
            assert np.equal(a, b).all()

        import matplotlib.pyplot as plt
        plt.spy(H1, markersize=.1)

        plt.show()

    else:
        from sigcom.coding.atsc import code_param_long
        codeParam = code_param_long.get([8, 15])
        H = make_pck(codeParam)
        isParityPermuted = True
        layerwise_pcks, diagOffsets = get_layerwise_pck(codeParam, isParityPermuted)
        PCM = layerwise_pcks_to_PCM(layerwise_pcks, codeParam)
        import matplotlib.pyplot as plt
        plt.spy(PCM, markersize=.1)
        plt.show()
