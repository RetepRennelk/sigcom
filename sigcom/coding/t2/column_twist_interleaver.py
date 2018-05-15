import numpy as np


class ColumnTwistInterleaver():
    def __init__(self, M, N_ldpc):
        self.M = M
        self.N_ldpc = N_ldpc
        tp =  {(16,64800):  [0,0,2,4,4,5,7,7],
               (16,16200):  [0,0,0,1,7,20,20,21],
               (64,64800):  [0,2,2,3,4,4,5,5,7,8,9],
               (64,16200):  [0,0,0,2,2,2,3,3,3,6,7,7],
               (256,64800): [0,2,2,2,2,3,7,15,16,20,22,22,27,27,28,32],
               (256,16200): [0,0,0,1,7,20,20,21]}
        self.tp = tp[(M,N_ldpc)]
        self.Nc = len(self.tp)

    def getPermVector(self):
        Nr = self.N_ldpc // self.Nc # Number of rows
        mBase = np.zeros((Nr, self.Nc))
        for col in range(self.Nc):
            mBase[:,col] = col*Nr+ (np.arange(Nr)-self.tp[col]) % Nr
        permVector = mBase.flatten()
        return permVector


if __name__ == '__main__':
    ctl = ColumnTwistInterleaver(16, 64800)
    pl = ctl.getPermVector()
    pl
        
