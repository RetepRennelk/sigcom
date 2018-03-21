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

    @staticmethod
    def average(rr1, rr2):
        MI0 = (rr1.MI0+rr2.MI0)/2
        MI1 = (rr1.MI1+rr2.MI1)/2
        MI0_1 = (rr1.MI0_1+rr2.MI0_1)/2
        MI1_0 = (rr1.MI1_0+rr2.MI1_0)/2
        MI_sum = (rr1.MI_sum+rr2.MI_sum)/2
        return RateRegion(MI0, MI1, MI0_1, MI1_0, MI_sum)


def plot_rate_region(C, I, N):
    plt.plot([0,np.log2(1+C/(I+N))],[np.log2(1+I/N)]*2,'b-')
    plt.plot([np.log2(1+C/(I+N)), np.log2(1+C/N)], [np.log2(1+I/N), np.log2(1+I/(C+N))], 'b-')
    plt.plot([np.log2(1+C/N)]*2, [np.log2(1+I/(C+N)),0], 'b-')
    R_min = np.min([np.log2(1+C/N), np.log2(1+I/N)])
    plt.plot([0,R_min],[0,R_min],'r--')
    plt.title('C={:.2g}, I={:.2g}, N={:.2g}'.format(C,I,N))
    plt.grid()
    plt.show()
