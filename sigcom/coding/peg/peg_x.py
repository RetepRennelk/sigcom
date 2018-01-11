import networkx as nx
import numpy as np
from networkx.algorithms.bipartite.matrix import biadjacency_matrix


class Peg_X():
    def __init__(self, M, N, dv, sw_rand):
        self.M = M
        self.N = N
        self.dv = dv
        self.sw_rand = sw_rand
        self.g = nx.Graph()
        self.init()

    def init(self):
        self.vn_nodes = -np.arange(1, self.N+1)
        self.cn_nodes = +np.arange(1, self.M+1)
        self.g.add_nodes_from(self.vn_nodes, bipartite=0)  # bit-nodes
        self.g.add_nodes_from(self.cn_nodes, bipartite=1)  # check-nodes

    def tosparse(self):
        return biadjacency_matrix(self.g, self.cn_nodes, self.vn_nodes)

    def run(self):
        for col in range(self.N):
            idx = self.find_cn(col, self.cn_nodes)
            self.g.add_edge(idx, -(col+1))
            for d in range(1, self.dv):
                cn_cands = self.bfs(col)
                idx = self.find_cn(col, cn_cands)
                self.g.add_edge(idx, -(col+1))

    def find_cn(self, col, rows):
        '''
        Find check-node with smallest degree
        '''
        h_sum = np.zeros(self.M)
        for i in range(col):
            h_sum[np.array(self.g[-(i+1)])-1] += 1
            # restrict candidates to rows
        h_sum = h_sum[rows-1]
        if self.sw_rand == 0:
            idx = np.argmin(h_sum)
        else:
            indices = np.where(h_sum == np.min(h_sum))[0]
            idx = np.random.choice(indices)
        return rows[idx]

    def bfs(self, col):
        l0 = np.array(list(self.g[-(col+1)]))
        l2 = np.array([], dtype=np.int32)
        while 1:
            l1 = np.array([])
            for x in l0:
                l1 = np.append(l1, list(self.g[x]))
            for x in np.unique(l1):
                l2 = np.append(l2, list(self.g[x]))
            l2 = np.unique(l2)
            N0 = l0.size
            N1 = l2.size
            if (N0 < self.M and N1 == self.M) or (N0 >= N1 and N0 < self.M):
                break
            l0 = l2.copy()
        cn_cands = np.setdiff1d(self.cn_nodes, l0)
        return cn_cands


if __name__ == '__main__':
    M = 10
    N = 10
    dv = 3
    peg_x = Peg_X(M, N, dv, sw_rand=1)
    peg_x.run()
    print(peg_x.tosparse().todense())
