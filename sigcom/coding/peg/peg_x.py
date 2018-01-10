import networkx as nx
import numpy as np
from networkx.algorithms.bipartite.matrix import biadjacency_matrix


def find_cn(col, rows, sw_rand=1):
    '''
    Find check-node with smallest degree
    '''
    h_sum = np.zeros(M)
    for i in range(col):
        h_sum[np.array(g[-(i+1)])-1] += 1
    # restrict candidates to rows
    h_sum = h_sum[rows-1]
    if sw_rand==0:
        idx = np.argmin(h_sum)
    else:
        indices = np.where(h_sum==np.min(h_sum))[0]
        idx = np.random.choice(indices)
    return rows[idx]


def bfs(col, dbg=0):
    l0 = np.array(list(g[-(col+1)]))
    l2 = np.array([], dtype=np.int32)
    while 1:
        l1 = np.array([])
        for x in l0:
            l1 = np.append(l1, list(g[x]))
        for x in np.unique(l1):
            l2 = np.append(l2, list(g[x]))
        l2 = np.unique(l2)
        N0 = l0.size
        N1 = l2.size
        if (N0 < M and N1 == M) or (N0 >= N1 and N0 < M):
            break
        l0 = l2.copy()
    cn_cands = np.setdiff1d(cn_nodes, l0)
    return cn_cands


M = 10
N = 10
dv = 3
g = nx.Graph()

vn_nodes = -np.arange(1, N+1)
cn_nodes = +np.arange(1, M+1)

g.add_nodes_from(vn_nodes, bipartite=0)  # bit-nodes
g.add_nodes_from(cn_nodes, bipartite=1)  # check-nodes

for col in range(N):
    idx = find_cn(col, cn_nodes)
    g.add_edge(idx, -(col+1))
    for d in range(1,dv):
        cn_cands = bfs(col)
        idx = find_cn(col, cn_cands)
        g.add_edge(idx, -(col+1))

mtx = biadjacency_matrix(g, cn_nodes, vn_nodes)
print(mtx.todense())
