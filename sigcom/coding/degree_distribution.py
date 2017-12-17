import numpy as np


def degree_distribution(H):
    N_edges = H.sum().sum()

    vns = np.squeeze(np.array(H.sum(axis=0), np.int32))
    vns_unique = np.unique(vns)
    vn_stats = np.array([])
    for vn in vns_unique:
        np.append((vn_stats, np.array([vn, vn*np.sum(vns == vn)/N_edges])))

    cns = np.squeeze(np.array(H.sum(axis=1), np.int32))
    cns_unique = np.unique(cns)
    cn_stats = np.array([])
    for cn in cns_unique:
        np.append((cn_stats, np.array([cn, cn*np.sum(cns == cn)/N_edges])))
    return vn_stats, cn_stats


if __name__ == '__main__':
    from sigcom.coding.atsc.code_param_short import get
    from sigcom.coding.PCM import PCM

    cp = get([8, 15])
    pcm = PCM(cp)

    H = pcm.make_layered(True)
    vn_stats, cn_stats = degree_distribution(H)

    print(vn_stats)
    print(' ')
    print(cn_stats)
