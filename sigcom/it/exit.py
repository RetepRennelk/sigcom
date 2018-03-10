import numpy as np
from sigcom.coding.atsc import code_param_long
from sigcom.coding.atsc import code_param_short
from sigcom.coding.PCM import PCM
from sigcom.coding.atsc.bititlv_long import bititlv_long
from sigcom.coding.atsc.bititlv_short import bititlv_short
from sigcom.coding.degree_distribution import degree_distribution_edge_view, degree_distribution_node_view
from sigcom.it.util import bits_to_apriori
from scipy.sparse import csr_matrix


def fill_PCM_apriori(N, K, indptr, indices, codebits, Ia):
    L = bits_to_apriori(codebits[indices], [Ia])
    return csr_matrix((L, indices, indptr), shape=(N-K, N))


def CN_exit_function(H, Ias):
    edge_stats = degree_distribution_edge_view(H)
    MIs = np.zeros(len(Ias))
    for k, Ia in enumerate(Ias):
        MI = 0
        for stats in edge_stats.cn:
            dc = stats[0]
            prob = stats[1]
            MI_temp = 1-getMutualInfo((dc-1)*getNoisePower([1-Ia]))[0]
            MI += prob*MI_tmp
        MIs[k] = MI
    return MIs


def demap_wrapper(fn, *args):
    def demap_callback(L_apriori):
        return fn(*args, L_apriori)
    return demap_callback


def VN_exit_function(H, Ias, codebits, demap_callback):
    MIs = np.zeros(len(Ias))
    N = H.shape[1]
    K = N-H.shape[0]
    for k, Ia in enumerate(Ias):
        VN = fill_PCM_apriori(N, K, H.indptr, H.indices, codebits, Ia)
        L_apri = VN.sum(axis=0).A1
        Llrs = demap_callback(L_apri)
        MI = mutual_information_magic(Llrs[H.indices]-VN-.data, codebits[H.indices], 1)
        MIs[k] = MI
    return MIs


class EXIT_mod_vn():
    '''
    Combined EXIT function of demapper and variable nodes
    '''
    def __init__(self, M, CR, N_ldpc):
        self.M = M
        self.ldM = int(np.log2(M))
        self.CR = CR
        self.N_ldpc = N_ldpc
        if N_ldpc == 16200:
            self.cp = code_param_short.get(CR)
            self.bil = bititlv_short(M, CR)
        elif N_ldpc == 64800:
            self.cp = code_param_long.get(CR)
            self.bil = bititlv_long(M, CR)
        self.pcm = PCM(self.cp)
        H = self.pcm.make_layered(True)
        self.edge_stats = degree_distribution_edge_view(H)
        self.node_stats = degree_distribution_node_view(H)
        self.make_vn_degrees_and_biled()

    def make_vn_degrees_and_biled(self):
        vn_degrees = np.zeros(self.cp.N, dtype=int)
        start = 0
        for stats in self.node_stats.vn:
            to = int(stats[1]*self.cp.N)
            vn_degrees[start:start+to] = stats[0]
            start += to
        self.vn_degrees = vn_degrees
        self.vn_degrees_biled = vn_degrees[self.bil]

    def encode_degrees_biled(self, m_degrees_biled):
        max_entry = np.max(m_degrees_biled)
        weights = [max_entry] * m_degrees_biled.shape[1]
        weights[0] = 1
        self.weights = np.cumprod(weights)
        return np.dot(m_degrees_biled, self.weights)

    def make_vn_degrees_combos(self):
        self.m_degrees_biled = self.vn_degrees_biled.reshape(-1, self.ldM)
        self.v_degree_combos_enc = self.encode_degrees_biled(self.m_degrees_biled)
        unique_degree_combos = np.unique(self.v_degree_combos_enc)
        vn_degree_combos = []
        for unique_degree_combo in unique_degree_combos:
            bidx = unique_degree_combo == self.v_degree_combos_enc
            idx = np.where(bidx)[0]
            vn_degree_combos.append([self.m_degrees_biled[idx[0]], np.mean(bidx)])
        self.vn_degree_combos = vn_degree_combos

    def map_demappers_to_vns(self):
        N_edges = np.sum(self.vn_degrees)
        for vdc in self.vn_degree_combos:
            vdc_enc = np.dot(vdc[0], self.weights)
            bidx = self.v_degree_combos_enc == vdc_enc
            subset = self.m_degrees_biled[bidx]
            edge_prob = []
            for i in range(self.ldM):
                edge_prob.append(np.sum(subset[:, i])/N_edges)
            vdc.append(edge_prob)


if __name__ == '__main__':
    M = 4
    CR = [8, 15]
    N_ldpc = 64800
    exit_mod_vn = EXIT_mod_vn(M, CR, N_ldpc)
