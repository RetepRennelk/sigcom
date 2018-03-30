import numpy as np


def make_noise(N):
    return 1/np.sqrt(2)*(np.random.randn(N) + 1j * np.random.randn(N))


def BEC_channel(N, p_err):
    return np.asarray(np.random.rand(N) > p_err, np.int)


def ricean_channel(N_cells, K_factor):
    phase = np.exp(1j*2*np.pi*np.random.rand(N_cells))
    noise = make_noise(N_cells)
    return np.sqrt(K_factor/(1+K_factor))*phase + noise/np.sqrt(1+K_factor)
