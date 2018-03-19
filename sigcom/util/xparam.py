import numpy as np
import _pickle
import gzip


class XParam():
    '''
    Use case: Average error statistics, e.g., BER
    '''
    def __init__(self):
        self.x = np.array([])
        self.stats = np.empty((0, 2))

    def _identify(self, x_new):
        num = x_new-self.x
        den = x_new+self.x
        eps = 1.e-16
        bidx = 4*(num/(den+eps))**2 < eps
        N = np.sum(bidx)
        assert N <= 1, "Index duplicate!"
        if N == 0:
            return False, -1
        else:
            return True, np.where(bidx)[0][0]

    def add(self, x_new, param):
        flag, idx = self._identify(x_new)
        if flag:
            self.stats[idx] += np.asarray(param)
        else:
            self.x = np.append(self.x, x_new)
            self.stats = np.vstack((self.stats, np.asarray(param)))
        a = np.argsort(self.x)
        self.x = self.x[a]
        self.stats = self.stats[a]

    def delete(self, x):
        flag, idx = self._identify(x)
        if flag:
            self.stats = np.delete(self.stats, idx, 0)
            self.x = np.delete(self.x, idx, 0)

    def save(self, filename):
        with gzip.open(filename, 'wb') as f:
            _pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with gzip.open(filename, 'rb') as f:
            loaded_object = _pickle.load(f)
        return loaded_object
