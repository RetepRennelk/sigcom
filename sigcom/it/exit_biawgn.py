from sigcom.it.exit import CN_exit_function
from sigcom.it.exit import VN_exit_function
from sigcom.it.exit_trace import EXIT_trace


class EXIT_BiAWGN():
    def __init__(self, H_dec, Ias):
        self.Ias = Ias
        self.H_dec = H_dec
        self.CN = CN_exit_function(H_dec, Ias)

    def make_VN(self, P_noise):
        self.VN = VN_exit_function(self.H_dec, self.Ias, P_noise)

    def plot_exit_trace(self):
        exit_trace = EXIT_trace(self.CN, self.Ias, self.Ias, self.VN)
        exit_trace.trace()
        exit_trace.plot()
