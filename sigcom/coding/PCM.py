from sigcom.coding.util import _pck_to_sparse_rows_and_cols
from sigcom.coding.util import make_pck as make_pck
from sigcom.coding.util import get_layerwise_pck, layerwise_pcks_to_PCM
from functools import lru_cache


class PCM(object):
    def __init__(self, code):
        self.code = code
        rows, cols = _pck_to_sparse_rows_and_cols(code)
        self.rows = rows
        self.cols = cols

    @lru_cache(maxsize=1)
    def make(self):
        return make_pck(self.code)

    @lru_cache(maxsize=2)
    def make_layered(self, isParityPermuted):
        layerwise_pcks, _ = get_layerwise_pck(self.code, isParityPermuted)
        return layerwise_pcks_to_PCM(layerwise_pcks, code)


if __name__ == '__main__':
    from sigcom.coding.atsc.pck_long import get_pck
    code = get_pck([8, 15])
    pcm = PCM(code)
