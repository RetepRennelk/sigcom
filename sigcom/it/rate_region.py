import numpy as np


class RateRegion():
    def __init__(self, MI0, MI1, MI0_1, MI1_0, MI_sum):
        self.MI0 = MI0
        self.MI1 = MI1
        self.MI0_1 = MI0_1
        self.MI1_0 = MI1_0
        self.MI_sum = MI_sum

    def print(self):
        print('MI0 = {:.3f}'.format(self.MI0))
        print('MI1 = {:.3f}'.format(self.MI1))
        print('MI0_1 = {:.3f}'.format(self.MI0_1))
        print('MI1_0 = {:.3f}'.format(self.MI1_0))
        print('MI_sum = {:.3f}'.format(self.MI_sum))

    def get(self):
        return self.MI0, self.MI1, self.MI0_1, self.MI1_0, self.MI_sum

    def plot(self, ax, linestyle, sw_main_diag=False):
        ax.plot([0,self.MI0],[self.MI1_0]*2,linestyle)
        ax.plot([self.MI0,self.MI0_1],[self.MI1_0,self.MI1],linestyle)
        ax.plot([self.MI0_1]*2,[0,self.MI1],linestyle)
        if sw_main_diag:
            MI_max = np.max((self.MI1_0, self.MI0_1))
            ax.plot([0, MI_max], [0, MI_max], linestyle)

    @staticmethod
    def average(rr1, rr2):
        MI0 = (rr1.MI0+rr2.MI0)/2
        MI1 = (rr1.MI1+rr2.MI1)/2
        MI0_1 = (rr1.MI0_1+rr2.MI0_1)/2
        MI1_0 = (rr1.MI1_0+rr2.MI1_0)/2
        MI_sum = (rr1.MI_sum+rr2.MI_sum)/2
        return RateRegion(MI0, MI1, MI0_1, MI1_0, MI_sum)


def plot_gaussian_rate_region(ax, C_I_dB, C_N_dB, linestyle):
    C_I = 10**(C_I_dB/10)
    C_N = 10**(C_N_dB/10)
    C = 1
    P_I = C/C_I
    N = C/C_N
    MI0 = np.log2(1+C/(P_I+N))
    MI1 = np.log2(1+P_I/(C+N))
    MI0_1 = np.log2(1+C/N)
    MI1_0 = np.log2(1+P_I/N)
    MI_sum = np.log2(1+(C+P_I)/N)
    rr = RateRegion(MI0, MI1, MI0_1, MI1_0, MI_sum)
    rr.plot(ax, linestyle)
    ax.grid()
    ax.set_title('C_I={}dB, C_N={}dB'.format(C_I_dB, C_N_dB))
