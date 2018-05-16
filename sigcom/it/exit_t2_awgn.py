import numpy as np
from sigcom.tx.modcod import ModCodT2
from sigcom.ch.util import make_noise
from sigcom.rx.util import demap
from sigcom.rx.softxor import softXorFwdBwd
from sigcom.it.util import mutual_information_magic


def update_CNs(H):
    M, N = H.shape
    for r in range(M):
        fr = H.indptr[r]
        to = H.indptr[r+1]
        H.data[fr:to] = softXorFwdBwd(H.data[fr:to])

M = 16
CR = [1,2]
N_ldpc = 64800
K = int(N_ldpc * CR[0] / CR[1])
N_codewords = 1
tx = ModCodT2(M, CR, N_ldpc)

Its = 50

SNR_dB = 6.
SNR = 10**(SNR_dB/10)
P_noise = 1/SNR

tx.generate(N_codewords)
N_fec_cells = len(tx.tx)
noise = make_noise(N_fec_cells)
rx = tx.tx + noise*np.sqrt(P_noise)

# Initialize
La = np.zeros(N_ldpc)
H_dec = tx.H_dec.copy()

H_dec.data[:] = 0

V2C = [0.0]
C2V = [0.0]

h = np.ones(N_fec_cells)

for it in range(Its):
    # Demap Upper Layer
    Llrs = demap(rx, tx.X, SNR, h, La)
    L_ext_dem = Llrs - La
    L_ext_dem_debil = np.zeros(N_ldpc)
    L_ext_dem_debil[tx.bil] = L_ext_dem

    # Update PCM
    L_ext_dem_debil += H_dec.sum(axis=0).A1
    H_dec.data = L_ext_dem_debil[H_dec.indices] - H_dec.data
    c = tx.m_codebits.flatten()
    MI = mutual_information_magic(H_dec.data, c[H_dec.indices], 1)
    V2C.append(MI)
    print('Demap:', np.sum((L_ext_dem_debil<0)!=tx.m_codebits.flatten()), MI)

    update_CNs(H_dec)
    MI = mutual_information_magic(H_dec.data, c[H_dec.indices], 1)
    C2V.append(MI)

    # Compute Extrinsic Info
    L_ext_fec = H_dec.sum(axis=0).A1
    L_ext_fec_bil = L_ext_fec[tx.bil].copy()
    La = L_ext_fec_bil

    N_errors = np.sum(((L_ext_fec+L_ext_dem_debil)<0) != tx.m_codebits.flatten())
    print('FEC UL:', N_errors, MI, it)
    if N_errors == 0:
        break

from sigcom.it.exit import CN_exit_function
CN_Ias = np.linspace(0.00001, .99999, 31)
C2Vs = CN_exit_function(tx.H_dec, CN_Ias)    

import matplotlib.pyplot as plt
f, ax = plt.subplots()
proxy = []
ax.step(C2V, V2C, 'b-')
ax.plot(C2V[:-1], V2C[1:], 'b-', linewidth=1)
ax.plot(C2Vs, CN_Ias, 'r-')
ax.set_title('SNR={}dB'.format(SNR_dB))
ax.set_xlabel('$MI_{A,VND}, MI_{E,CND}$')
ax.set_ylabel('$MI_{E,VND}, MI_{A,CND}$')
plt.grid()
plt.show()

