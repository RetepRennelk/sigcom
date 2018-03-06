import numpy as np
from collections import namedtuple


def degree_distribution_edge_view(H):
    N_edges = H.sum().sum()

    vns = np.squeeze(np.array(H.sum(axis=0), np.int32))
    vns_unique = np.unique(vns)[::-1]
    vn_stats = [np.array([vn, vn*np.sum(vns == vn)/N_edges])
                for vn in vns_unique]

    cns = np.squeeze(np.array(H.sum(axis=1), np.int32))
    cns_unique = np.unique(cns)[::-1]
    cn_stats = [np.array([cn, cn*np.sum(cns == cn)/N_edges])
                for cn in cns_unique]

    edge_stats = namedtuple('edge_stats', 'vn cn')
    return edge_stats(np.array(vn_stats), np.array(cn_stats))


def degree_distribution_node_view(H):
    vns = np.squeeze(np.array(H.sum(axis=0), np.int32))
    vns_unique = np.unique(vns)[::-1]
    vn_stats = [np.array([vn, np.mean(vns == vn)])
                for vn in vns_unique]

    cns = np.squeeze(np.array(H.sum(axis=1), np.int32))
    cns_unique = np.unique(cns)[::-1]
    cn_stats = [np.array([cn, np.mean(cns == cn)])
                for cn in cns_unique]

    node_stats = namedtuple('node_stats', 'vn cn')
    return node_stats(np.array(vn_stats), np.array(cn_stats))


def edge_to_node_view(edge_stats):
    vn_node_stats = edge_stats.vn[:, 1] / edge_stats.vn[:, 0]
    vn_node_stats = vn_node_stats/vn_node_stats.sum()
    vn_node_stats = np.vstack((edge_stats.vn[:, 0], vn_node_stats)).T

    cn_node_stats = edge_stats.cn[:, 1] / edge_stats.cn[:, 0]
    cn_node_stats = cn_node_stats/cn_node_stats.sum()
    cn_node_stats = np.vstack((edge_stats.cn[:, 0], cn_node_stats)).T

    node_stats = namedtuple('node_stats', 'vn cn')
    return node_stats(vn_node_stats, cn_node_stats)


def node_to_edge_view(node_stats):
    vn_edge_stats = node_stats.vn[:, 1] * node_stats.vn[:, 0]
    vn_edge_stats = vn_edge_stats/vn_edge_stats.sum()
    vn_edge_stats = np.vstack((node_stats.vn[:, 0], vn_edge_stats)).T

    cn_edge_stats = node_stats.cn[:, 1] * node_stats.cn[:, 0]
    cn_edge_stats = cn_edge_stats/cn_edge_stats.sum()
    cn_edge_stats = np.vstack((node_stats.cn[:, 0], cn_edge_stats)).T

    edge_stats = namedtuple('edge_stats', 'vn cn')
    return edge_stats(vn_edge_stats, cn_edge_stats)


if __name__ == '__main__':
    from sigcom.coding.atsc.code_param_long import get
    from sigcom.coding.PCM import PCM

    cp = get([8, 15])
    pcm = PCM(cp)

    H = pcm.make_layered(True)
    edge_stats = degree_distribution_edge_view(H)
    node_stats = degree_distribution_node_view(H)

    node_stats2 = edge_to_node_view(edge_stats)
    edge_stats2 = node_to_edge_view(node_stats)

    print('edge_stats.vn\n', edge_stats.vn)
    print(' ')
    print('edge_stats.cn\n', edge_stats.cn)
    print(' ')
    print('node_stats.vn\n', node_stats.vn)
    print(' ')
    print('node_stats.cn\n', node_stats.cn)
    print('----------------------------------------')
    print('edge_stats2.vn\n', edge_stats.vn - edge_stats2.vn)
    print(' ')
    print('edge_stats2.cn\n', edge_stats.cn - edge_stats2.cn)
    print(' ')
    print('node_stats2.vn\n', node_stats.vn - node_stats2.vn)
    print(' ')
    print('node_stats2.cn\n', node_stats.cn - node_stats2.cn)
