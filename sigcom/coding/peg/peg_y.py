from ipdb import set_trace as bp
import networkx as nx
import numpy as np
from networkx.algorithms.bipartite.matrix import biadjacency_matrix


class Peg_Y():
    def __init__(self, M, N, dv):
        self.M = M
        self.N = N
        self.dv = dv
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
        for vn in self.vn_nodes:
            idx = self.find_cn(vn, self.cn_nodes)
            self.g.add_edge(idx, vn)
            for d in range(1, self.dv):
                cn_cands, ext_sw = self.bfs(vn)
                if ext_sw and len(cn_cands) > 1:
                    cn = self.test_cycle(vn, cn_cands)
                else:
                    cn = self.find_cn(vn, cn_cands)
                self.g.add_edge(cn, vn)

    def test_cycle(self, vn, cn_cands):
        '''
        Add edge tentatively and check length of cycle being 
        created. Choose the edge which creates the longest cycle.
        '''
        cycles = []
        for cn in cn_cands:
            g_tmp = self.g.copy()
            g_tmp.add_edge(cn, vn)
            cycles.append(nx.find_cycle(g_tmp, vn))
        cycle_len = np.array([len(c) for c in cycles])
        max_cycle_len = np.max(cycle_len)
        indices = np.where(cycle_len == max_cycle_len)[0]
        idx = np.random.choice(indices)
        return cn_cands[idx]

    def find_cn(self, vn, cns):
        '''
        Find check-node with smallest degree
        '''
        h_sum = np.zeros(self.M)
        for v in self.vn_nodes:
            if v == vn:
                break
            h_sum[np.array(self.g[v])-1] += 1
        # restrict candidates to rows
        h_sum = h_sum[cns-1]
        indices = np.where(h_sum == np.min(h_sum))[0]
        idx = np.random.choice(indices)
        return cns[idx]

    def bfs(self, vn):
        l0 = np.array(list(self.g[vn]))
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
            ext_sw = N0 < self.M and N1 == self.M
            if ext_sw or (N0 >= N1 and N0 < self.M):
                break
            l0 = l2.copy()
        cn_cands = np.setdiff1d(self.cn_nodes, l0)
        return cn_cands, ext_sw

    def cycles(self):
        vn_cycles = [nx.find_cycle(self.g, vn) for vn in self.vn_nodes]
        cn_cycles = [nx.find_cycle(self.g, cn) for cn in self.cn_nodes]
        print('vn_cycles', vn_cycles)
        print('cn_cycles', cn_cycles)
        len_vn_cycles = [len(c) for c in vn_cycles]
        len_cn_cycles = [len(c) for c in cn_cycles]
        print('len_vn_cycles', len_vn_cycles)
        print('len_cn_cycles', len_cn_cycles)


if __name__ == '__main__':
    M = 4
    N = 8
    dv = 2
    peg_y = Peg_Y(M, N, dv)
    peg_y.run()
    print(peg_y.tosparse().todense())
    peg_y.cycles()
