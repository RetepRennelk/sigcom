import numpy as np
from sigcom.tx.util import generate_bits, qam_alphabet, map_bits_to_symbol_alphabet
from sigcom.coding.atsc import bititlv_short, bititlv_long
from sigcom.coding.atsc import code_param_long
from sigcom.coding.PCM import PCM
from sigcom.coding.util import get_parity_interleaver


class ModCodAtsc():
    def __init__(self, M, CR, N_ldpc):
        self.M = M
        self.CR = CR
        self.N_ldpc = N_ldpc

        if M == 4:
            self.X = qam_alphabet(M)
        else:
            assert False, "Implement me!"

        if N_ldpc == 16200:
            self.bil = bititlv_short.bititlv_short(M, CR)
        else:
            self.bil = bititlv_long.bititlv_long(M, CR)

        self.cp = code_param_long.get(CR)
        self.pcm = PCM(self.cp)
        self.H_enc = self.pcm.make()
        self.H_dec = self.pcm.make_layered(True)

        self.parintl = get_parity_interleaver(self.cp.K)

    def generate(self, N_codewords):
        self.bits = generate_bits(N_codewords*self.cp.K)
        self.m_bits = self.bits.reshape(-1, self.cp.K).T
        parity = np.int64(self.H_enc[:,:self.cp.K].dot(self.m_bits))
        parity = np.cumsum(parity, axis=0) % 2
        self.m_codebits = np.vstack((self.m_bits, parity))[self.parintl, :]
        self.m_codebits_biled = self.m_codebits[self.bil,:]
        c = self.m_codebits_biled.T.flatten()
        self.tx = map_bits_to_symbol_alphabet(c, self.X)
