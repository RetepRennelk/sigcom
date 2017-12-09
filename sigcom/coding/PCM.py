from sigcom.coding.util import _pck_to_sparse_rows_and_cols
from sigcom.coding.util import make_pck as make_pck
from sigcom.coding.util import get_layerwise_pck, layerwise_pcks_to_PCM
from functools import lru_cache


class PCM(object):
    def __init__(self, codeParam):
        self.codeParam = codeParam

    @lru_cache(maxsize=1)
    def pck_to_sparse_rows_and_cols(self):
        rows, cols = _pck_to_sparse_rows_and_cols(self.codeParam)
        return rows, cols

    @lru_cache(maxsize=1)
    def make(self):
        return make_pck(self.codeParam)

    @lru_cache(maxsize=2)
    def make_layered(self, isParityPermuted):
        layerwise_pcks, _ = get_layerwise_pck(self.codeParam, isParityPermuted)
        return layerwise_pcks_to_PCM(layerwise_pcks, self.codeParam)


if __name__ == '__main__':
    from sigcom.coding.atsc.pck_long import get_code_param
    codeParam = get_code_param([8, 15])
    pcm = PCM(codeParam)
