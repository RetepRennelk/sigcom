from sigcom.tx.util import qam_alphabet


class ModCod():
    def __init__(self, M, CR, N_ldpc):
        self.M = M
        self.CR = CR
        self.N_ldpc = N_ldpc

        if M == 4:
            self.Q = qam_alphabet(M)
        else:
            assert False, "Implement me!"
