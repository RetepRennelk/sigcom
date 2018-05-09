import numpy as np
from sigcom.tx.modcod import ModCodSP1p4_rs
from sigcom.ch.util import make_noise
from sigcom.rx.spm_rs import demap0, demap1
from sigcom.rx.softxor import softXorFwdBwd
from sigcom.it.util import mutual_information_magic


def update_CNs(H):
    M, N = H.shape
    for r in range(M):
        fr = H.indptr[r]
        to = H.indptr[r+1]
        H.data[fr:to] = softXorFwdBwd(H.data[fr:to])

M = 4
CR = [8, 15]
N_ldpc = 64800
K = int(N_ldpc*CR[0]/CR[1])
N_codewords = 1
tx = ModCodSP1p4_rs(M, CR, N_ldpc)
tx.tx1.bil = tx.tx1.bil[::-1]
tx.update(N_codewords)

C_N_dB = 5
rho_dB = 3
C_I_dB = 0

C_N = 10**(C_N_dB/10)
C_I = 10**(C_I_dB/10)
rho = 10**(rho_dB/10)

Cp = 2*rho/(1+rho)
Cpp = 2/(1+rho)
Ip = 2/(1+rho)/C_I
Ipp = 2*rho/(1+rho)/C_I
P_noise = 1/C_N

Powers0 = [Cp, Cpp]
Powers1 = [Ip, Ipp]

tx.generate(Powers0, Powers1)
N_fec_cells = len(tx.tx)
noise = make_noise(N_fec_cells)
rx = tx.tx + noise*np.sqrt(P_noise)

# Initialize
La0 = np.zeros(N_ldpc)
La1 = np.zeros(N_ldpc)
H0 = tx.tx0.H_dec.copy()
H1 = tx.tx1.H_dec.copy()
H0.data[:]=0
H1.data[:]=0

V2C0 = [0.0]
C2V0 = [0.0]
V2C1 = [0.0]
C2V1 = [0.0]

Its = 50
for it in range(Its):
    # Demap Upper Layer
    Llrs0 = demap0(rx, tx.tx0.X, tx.tx1.X, Powers0, Powers1, tx.phase, P_noise, N_fec_cells, La0, La1)
    L_ext_dem0 = Llrs0 - La0
    L_ext_dem_debil0 = np.zeros(N_ldpc)
    L_ext_dem_debil0[tx.tx0.bil] = L_ext_dem0
    MI = mutual_information_magic(L_ext_dem_debil0, tx.tx0.m_codebits.flatten(), 1)
    V2C0.append(MI)
    print('Demap UL:', np.sum((L_ext_dem_debil0<0)!=tx.tx0.m_codebits.flatten()), MI)

    # Update PCM
    L_ext_dem_debil0 += H0.sum(axis=0).A1
    H0.data = L_ext_dem_debil0[H0.indices] - H0.data
    update_CNs(H0)

    # Compute Extrinsic Info
    L_ext_fec0 = H0.sum(axis=0).A1
    L_ext_fec_bil0 = L_ext_fec0[tx.tx0.bil].copy()
    La0 = L_ext_fec_bil0
    MI = mutual_information_magic(L_ext_fec0, tx.tx0.m_codebits.flatten(), 1)
    C2V0.append(MI)
    N_errors = np.sum(((L_ext_fec0+L_ext_dem_debil0)<0) != tx.tx0.m_codebits.flatten())
    print('FEC UL:', N_errors, MI)
    if N_errors == 0:
        break

    # Demap Lower Layer
    Llrs1 = demap1(rx, tx.tx0.X, tx.tx1.X, Powers0, Powers1, tx.phase, P_noise, N_fec_cells, La0, La1)
    L_ext_dem1 = Llrs1 - La1
    L_ext_dem_debil1 = np.zeros(N_ldpc)
    L_ext_dem_debil1[tx.tx1.bil] = L_ext_dem1
    MI = mutual_information_magic(L_ext_dem_debil1, tx.tx1.m_codebits.flatten(), 1)
    V2C1.append(MI)
    print('Demap LL:',np.sum((L_ext_dem1<0)!=tx.tx1.codebits), MI)

    # Update
    L_ext_dem_debil1 += H1.sum(axis=0).A1
    H1.data = L_ext_dem_debil1[H1.indices] - H1.data
    update_CNs(H1)

    L_ext_fec1 = H1.sum(axis=0).A1
    L_ext_fec_bil1 = L_ext_fec1[tx.tx1.bil].copy()
    La1 = L_ext_fec_bil1
    MI = mutual_information_magic(L_ext_fec1, tx.tx1.m_codebits.flatten(), 1)
    C2V1.append(MI)
    print('FEC LL:',np.sum(((L_ext_fec1+L_ext_dem_debil1)<0) != tx.tx1.m_codebits.flatten()), MI)

y = []
for v2c in V2C0:
    y.extend([v2c, v2c])
x = []
for c2v in C2V0:
    x.extend([c2v, c2v])

import matplotlib.pyplot as plt
plt.plot(x[:-1],y[1:])
plt.show()

