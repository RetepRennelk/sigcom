import numpy as np


def rrcosfilter(N, alpha, Ts, Fs):

    T_delta = 1/float(Fs)
    h_rrc = np.zeros(N, dtype=float)

    for x in np.arange(N):
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                    4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                    (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)

    time_idx = ((np.arange(N)-N/2))*T_delta

    return h_rrc, time_idx

if __name__ == '__main__':
    N = 1000*2
    alpha = .5
    Ts = 1
    Fs = 100
    h, k = rrcosfilter(N, alpha, Ts, Fs)
    import matplotlib.pyplot as plt

    plt.plot(k, h, 'b.-')
    plt.grid()
    plt.show()
