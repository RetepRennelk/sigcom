import numpy as np


def degree_distribution_edge_view(H):
    N_edges = H.sum().sum()

    vns = np.squeeze(np.array(H.sum(axis=0), np.int32))
    vns_unique = np.unique(vns)
    vn_stats = [np.array([vn, vn*np.sum(vns == vn)/N_edges])
                for vn in vns_unique]

    cns = np.squeeze(np.array(H.sum(axis=1), np.int32))
    cns_unique = np.unique(cns)
    cn_stats = [np.array([cn, cn*np.sum(cns == cn)/N_edges])
                for cn in cns_unique]

    return np.array(vn_stats), np.array(cn_stats)


def degree_distribution_node_view(H):
    vns = np.squeeze(np.array(H.sum(axis=0), np.int32))
    vns_unique = np.unique(vns)
    vn_stats = [np.array([vn, np.mean(vns == vn)])
                for vn in vns_unique]

    cns = np.squeeze(np.array(H.sum(axis=1), np.int32))
    cns_unique = np.unique(cns)
    cn_stats = [np.array([cn, np.mean(cns == cn)])
                for cn in cns_unique]

    return np.array(vn_stats), np.array(cn_stats)


def edge_to_node_view(vn_edge_stats, cn_edge_stats):
    vn_node_stats = vn_edge_stats[:, 1] / vn_edge_stats[:, 0]
    vn_node_stats = vn_node_stats/vn_node_stats.sum()
    vn_node_stats = np.vstack((vn_edge_stats[:, 0], vn_node_stats)).T

    cn_node_stats = cn_edge_stats[:, 1] / cn_edge_stats[:, 0]
    cn_node_stats = cn_node_stats/cn_node_stats.sum()
    cn_node_stats = np.vstack((cn_edge_stats[:, 0], cn_node_stats)).T

    return vn_node_stats, cn_node_stats


def node_to_edge_view(vn_node_stats, cn_node_stats):
    vn_edge_stats = vn_node_stats[:, 1] * vn_node_stats[:, 0]
    vn_edge_stats = vn_edge_stats/vn_edge_stats.sum()
    vn_edge_stats = np.vstack((vn_node_stats[:, 0], vn_edge_stats)).T

    cn_edge_stats = cn_node_stats[:, 1] * cn_node_stats[:, 0]
    cn_edge_stats = cn_edge_stats/cn_edge_stats.sum()
    cn_edge_stats = np.vstack((cn_node_stats[:, 0], cn_edge_stats)).T

    return vn_edge_stats, cn_edge_stats


if __name__ == '__main__':
    from sigcom.coding.atsc.code_param_long import get
    from sigcom.coding.PCM import PCM

    cp = get([8, 15])
    pcm = PCM(cp)

    H = pcm.make_layered(True)
    vn_edge_stats, cn_edge_stats = degree_distribution_edge_view(H)
    vn_node_stats, cn_node_stats = degree_distribution_node_view(H)
    
    vn_node_stats2, cn_node_stats2 = edge_to_node_view(vn_edge_stats, cn_edge_stats)
    vn_edge_stats2, cn_edge_stats2 = node_to_edge_view(vn_node_stats, cn_node_stats)

    print('vn_edge_stats\n', vn_edge_stats)
    print(' ')
    print('cn_edge_stats\n', cn_edge_stats)
    print(' ')
    print('vn_node_stats\n', vn_node_stats)
    print(' ')
    print('cn_node_stats\n', cn_node_stats)
    print('----------------------------------------')
    print('vn_edge_stats2\n', vn_edge_stats - vn_edge_stats2)
    print(' ')
    print('cn_edge_stats2\n', cn_edge_stats - cn_edge_stats2)
    print(' ')
    print('vn_node_stats2\n', vn_node_stats - vn_node_stats2)
    print(' ')
    print('cn_node_stats2\n', cn_node_stats - cn_node_stats2)
