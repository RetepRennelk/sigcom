from scipy.sparse import csc_matrix, csr_matrix
import numpy as np

def _pck_to_sparse_rows_and_cols(code):
    '''
    Take the definition of the parity check matrices from DVB or ATSC 
    and translate them to a list representation of all edges.
    Call 'H = csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(N-K,N))'
    afterwards to get a sparse matrix representation.
    '''
    nLayers = int((code['N']-code['K'])/code['nCyclicFactor'])
    rows = []
    cols = []
    col = 0
    for row in code['pck']:
        for m in range(code['nCyclicFactor']):
            for p in row:
                idx = (p+m*nLayers) % (code['N']-code['K'])
                rows.append(idx)
                cols.append(col)
            col+=1
    for k in range(code['N']-code['K']-1):
        rows.extend([k, k+1])
        cols.extend([col, col])
        col += 1
    rows.append(code['N']-code['K']-1)
    cols.append(col)
    return rows, cols

def make_pck(code):
    rows, cols = _pck_to_sparse_rows_and_cols(code)
    shape = (code['N']-code['K'], code['N'])
    H = csc_matrix((np.ones(len(rows)), (rows,cols)), shape=shape)
    return H
    
def get_parity_interleaver(K, N = 64800, nCyclicFactor = 360):
    parity = np.arange(N-K).reshape((nCyclicFactor,-1)).T.reshape(-1)
    return np.hstack((np.arange(K), parity+K))
    
def get_layerwise_pck(code, isParityPermuted):
    '''
    layerwise_pcks, diagOffsets = get_layerwise_pck(code, isParityPermuted)
    '''
    layerwise_pcks = []
    N_layers = int((code['N']-code['K'])/code['nCyclicFactor'])
    diagOffsets = np.zeros(N_layers)    
    for layer in range(N_layers):
        pck = []
        for segment, ParityChecks in enumerate(code['pck']):
            ParityChecks = np.array(ParityChecks)
            CurrentLayers = (ParityChecks - layer) % N_layers
            VerticalShifts = (ParityChecks[CurrentLayers == 0] - layer) / N_layers
            if len(VerticalShifts) > 0:
                DiagonalShifts = (code['nCyclicFactor']-VerticalShifts) % code['nCyclicFactor']
                ColumnOffset = segment*code['nCyclicFactor']
                pck.extend([[ColumnOffset,int(ds)] for ds in DiagonalShifts])

        if isParityPermuted:
            pck.append([code['K']+code['nCyclicFactor']*layer, 0])
            if layer == 0:
                pck.append([code['N']-code['nCyclicFactor'], code['nCyclicFactor']-1])
            else:
                pck.append([code['K']+code['nCyclicFactor']*(layer-1), 0])
        else:
            pck.append([code['K'], layer])
            pck.append([code['K'], layer-1])
        layerwise_pcks.append(pck)

        if layer+1 < N_layers:
            diagOffsets[layer+1] = int(diagOffsets[layer] + len(pck))
    return layerwise_pcks, diagOffsets
    
def layerwise_pcks_to_PCM(layerwise_pcks, code):
    '''
    run 'layerwise_pcks, diagOffsets = get_layerwise_pck(code, isParityPermuted)'
    '''
    cols = []
    rows = []
    row = 0
    for i, pck in enumerate(layerwise_pcks):
        pck = np.array(pck)
        for m in range(code['nCyclicFactor']):
            x = pck[:,0] + (pck[:,1]+m) % code['nCyclicFactor']
            if i==0 and m==0:
                x = x[:-1]
            cols.extend(x)
            rows.extend([row]*len(x))
            row += 1
            
    shape = (code['N']-code['K'], code['N'])
    PCM = csr_matrix((np.ones(len(rows)), (rows,cols)), shape=shape)
    return PCM
