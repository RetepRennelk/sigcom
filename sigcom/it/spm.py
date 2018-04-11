import numpy as np
from sigcom.rx.util import _max_star
from sigcom.ch.util import make_noise
from numba import njit
from sigcom.it.rate_region import RateRegion
from sigcom.tx.spm import SP1p4


@njit
def _MI0(rx, tx0, Ps, X0, X1, h0, h1, P_noise):
    N_cells = len(rx)
    P0 = Ps[0]
    P1 = Ps[1]
    M0 = len(X0)
    MI = 0
    for k in range(N_cells):
        num = -np.inf
        den = -np.inf
        for x1 in X1:
            D = rx[k] - np.sqrt(P0)*tx0[k]*h0[k] - np.sqrt(P1)*x1*h1[k]
            num = _max_star(num, -np.abs(D)**2/P_noise)
            for x0 in X0:
                D = rx[k] - np.sqrt(P0)*x0*h0[k] - np.sqrt(P1)*x1*h1[k]
                den = _max_star(den, -np.abs(D)**2/P_noise)
        MI += num - den
    return np.log2(M0)+MI/N_cells/np.log(2.0)


@njit
def _MI1(rx, tx1, Ps, X0, X1, h0, h1, P_noise):
    N_cells = len(rx)
    P0 = Ps[0]
    P1 = Ps[1]
    M0 = len(X0)
    MI = 0
    for k in range(N_cells):
        num = -np.inf
        den = -np.inf
        for x0 in X0:
            D = rx[k] - np.sqrt(P0)*x0*h0[k] - np.sqrt(P1)*tx1[k]*h1[k]
            num = _max_star(num, -np.abs(D)**2/P_noise)
            for x1 in X1:
                D = rx[k] - np.sqrt(P0)*x0*h0[k] - np.sqrt(P1)*x1*h1[k]
                den = _max_star(den, -np.abs(D)**2/P_noise)
        MI += num - den
    return np.log2(M0)+MI/N_cells/np.log(2.0)


@njit
def _MI0_1(rx, tx0, tx1, Ps, X0, X1, h0, h1, P_noise):
    N_cells = len(rx)
    P0 = Ps[0]
    P1 = Ps[1]
    M0 = len(X0)
    MI = 0
    for k in range(N_cells):
        den = -np.inf
        D = rx[k] - np.sqrt(P0)*tx0[k]*h0[k] - np.sqrt(P1)*tx1[k]*h1[k]
        num = -np.abs(D)**2/P_noise
        for x0 in X0:
            D = rx[k] - np.sqrt(P0)*x0*h0[k] - np.sqrt(P1)*tx1[k]*h1[k]
            den = _max_star(den, -np.abs(D)**2/P_noise)
        MI += num - den
    return np.log2(M0)+MI/N_cells/np.log(2.0)


@njit
def _MI1_0(rx, tx0, tx1, Ps, X0, X1, h0, h1, P_noise):
    N_cells = len(rx)
    P0 = Ps[0]
    P1 = Ps[1]
    M1 = len(X1)
    MI = 0
    for k in range(N_cells):
        den = -np.inf
        D = rx[k] - np.sqrt(P0)*tx0[k]*h0[k] - np.sqrt(P1)*tx1[k]*h1[k]
        num = -np.abs(D)**2/P_noise
        for x1 in X1:
            D = rx[k] - np.sqrt(P0)*tx0[k]*h0[k] - np.sqrt(P1)*x1*h1[k]
            den = _max_star(den, -np.abs(D)**2/P_noise)
        MI += num - den
    return np.log2(M1)+MI/N_cells/np.log(2.0)

@njit
def _MI_sum(rx, tx0, tx1, Ps, X0, X1, h0, h1, P_noise):
    N_cells = len(rx)
    P0 = Ps[0]
    P1 = Ps[1]
    M0 = len(X0)
    M1 = len(X1)
    MI = 0
    for k in range(N_cells):
        D = rx[k] - np.sqrt(P0)*tx0[k]*h0[k] - np.sqrt(P1)*tx1[k]*h1[k]
        num = -np.abs(D)**2/P_noise
        den = -np.inf
        for x0 in X0:
            for x1 in X1:
                D = rx[k] - np.sqrt(P0)*x0*h0[k] - np.sqrt(P1)*x1*h1[k]
                den = _max_star(den, -np.abs(D)**2/P_noise)
        MI += num - den
    return np.log2(M0)+np.log2(M1)+MI/N_cells/np.log(2.0)


class MI_SP1p4():
    def __init__(self, sp1p4):
        self.sp1p4 = sp1p4
        self.update()

    def update(self):
        self.noise = make_noise(self.sp1p4.N_cells)

    def run(self, P_noise):
        tx0 = self.sp1p4.tx0
        tx1 = self.sp1p4.tx1
        Ps = self.sp1p4.Ps
        X0 = self.sp1p4.X0
        X1 = self.sp1p4.X1
        phase = self.sp1p4.phase

        rx = self.sp1p4.tx + np.sqrt(P_noise)*self.noise

        MI0 = _MI0(rx, tx0, Ps, X0, X1, np.ones(len(self)), phase, P_noise)
        MI1 = _MI1(rx, tx1, Ps, X0, X1, np.ones(len(self)), phase, P_noise)
        MI0_1 = _MI0_1(rx, tx0, tx1, Ps, X0, X1, np.ones(len(self)), phase, P_noise)
        MI1_0 = _MI1_0(rx, tx0, tx1, Ps, X0, X1, np.ones(len(self)), phase, P_noise)
        MI_sum = _MI_sum(rx, tx0, tx1, Ps, X0, X1, np.ones(len(self)), phase, P_noise)

        self.rate_region = RateRegion(MI0, MI1, MI0_1, MI1_0, MI_sum)

    def print(self):
        self.rate_region.print()

    def get(self):
        return self.rate_region.get()

    def get_rate_region(self):
        return self.rate_region


class Average_MI_SP1p4():
    def __init__(self, N_cells):
        self.N_cells = N_cells
        self.sp1p4 = SP1p4(N_cells)
        self.mi_sp1p4 = MI_SP1p4(self.sp1p4)
        self.update()

    def update(self):
        self.sp1p4.update()
        self.mi_sp1p4.update()

    def run(self, P1p, P_noise):
        self.sp1p4.generate([P1p, 2-P1p])
        self.mi_sp1p4.run(P_noise)
        rr_p = self.mi_sp1p4.get_rate_region()
        self.sp1p4.generate([2-P1p, P1p])
        self.mi_sp1p4.run(P_noise)
        rr_pp = self.mi_sp1p4.get_rate_region()
        self.MI0 = (rr_p.MI0 + rr_pp.MI0_1)/2
        self.MI1 = (rr_p.MI1_0 + rr_pp.MI1)/2
        self.rate_region = RateRegion.average(rr_p, rr_pp)
