import numpy as np
from sigcom.tx.util import qam_alphabet, generate_bits, \
    map_bits_to_symbol_alphabet, make_cells
from sigcom.ch.util import make_noise
from sigcom.rx.util import demap
from sigcom.it.util import mutual_information_magic
import matplotlib.pyplot as plt


class QamAwgnBicmCapacity():
    def __init__(self, qam, N_cells):
        self.qam = qam
        self.N_cells = N_cells
        M = len(qam)
        self.ldM = int(np.log2(M))
        self.update()

    def update(self):
        self.tx, self.bits = make_cells(self.qam, self.N_cells)
        self.noise = make_noise(self.N_cells)

    def _generate_mi(self, SNR_dB):
        SNR = 10**(SNR_dB/10)
        rx = self.tx+self.noise/np.sqrt(SNR)
        h = np.ones(self.N_cells)
        Llrs = demap(rx, self.qam, SNR, h)
        MI = np.sum(mutual_information_magic(Llrs, self.bits, self.ldM))
        return MI

    def compute(self, SNRs_dB):
        MIs = []
        for SNR_dB in SNRs_dB:
            MI = self._generate_mi(SNR_dB)
            MIs.append(MI)
        return MIs


class QamBicmCapacity():
    '''
    Compute BICM capacity for fading channels

    channel = QamBicmCapacity.wrap(ricean_channel, N_cells, K_factor)
    q = QamBicmCapacity(qam, N_cells, channel)
    '''
    def __init__(self, qam, N_cells, channel):
        self.qam = qam
        self.N_cells = N_cells
        self.channel = channel
        M = len(qam)
        self.ldM = int(np.log2(M))
        self.update()

    def update(self):
        self.tx, self.bits = make_cells(self.qam, self.N_cells)
        self.h = self.channel()
        self.noise = make_noise(self.N_cells)

    def _generate_mi(self, SNR_dB):
        SNR = 10**(SNR_dB/10)
        rx = self.h*self.tx+self.noise/np.sqrt(SNR)
        Llrs = demap(rx, self.qam, SNR, self.h)
        MI = np.sum(mutual_information_magic(Llrs, self.bits, self.ldM))
        return MI

    def compute(self, SNRs_dB):
        MIs = []
        for SNR_dB in SNRs_dB:
            MI = self._generate_mi(SNR_dB)
            MIs.append(MI)
        return MIs

    @staticmethod
    def wrap(func, *args):
        '''
        Use this function to call a channel generator like ricean_channel.
        '''
        def wrapped():
            return func(*args)
        return wrapped


if __name__ == '__main__':
    if 1:
        ldM = 2
        M = 2**ldM
        N_cells = 100000
        qam = qam_alphabet(M)
        c = QamAwgnBicmCapacity(qam, N_cells)
        SNRs_dB = np.linspace(-10,21,21)
        MIs = c.compute(SNRs_dB)
    else:
        ldM = 2
        M = 2**ldM
        N_cells = 100000
        qam = qam_alphabet(M)
        tx, bits = make_cells(qam, N_cells)
        noise = make_noise(N_cells)
        SNRs_dB = np.linspace(-10,21,21)
        h = np.ones(N_cells)
        MIs = []
        for SNR in 10**(SNRs_dB/10):
            rx = tx+noise/np.sqrt(SNR)
            Llrs = demap(rx, qam, SNR, h)
            MI = np.sum(mutual_information_magic(Llrs, bits, ldM))
            MIs.append(MI)

    plt.plot(SNRs_dB, MIs, 'bo-')
    plt.xticks(np.arange(-10,20+1,1))
    plt.grid()
    plt.show()
