import numpy as np
import pandas as pd

class Ising_Knockoffs:
    
    def __init__(self, Z, Theta):
        self.Theta = Theta
        self.ncol = Theta.shape[1]
        self.Z = Z        
        
    def __energy(self, i, z_i, z, zt):
        e = 0
        for j in range(self.ncol):
            if j != i:
                e += z_i * zt[j] * self.Theta[i, j]
        for j in range(self.ncol):
            e += z_i * z[j] * self.Theta[i, j]

        return(np.exp(e))
    
    def __predict_sample(self, i, z, zt):
        p_one = self.__energy(i=i, z=z, zt=zt, z_i=1)
        p_minus_one = self.__energy(i=i, z=z, zt=zt, z_i=-1)
        u = np.random.uniform()
        prob = p_minus_one / (p_one + p_minus_one)
        if u <= prob:
            return(-1)
        else:
            return(1)
         
    def sample_row(self, z, zt=None):
        if zt is None:
            zt=np.random.choice(a=[-1, 1], replace=True, size=self.ncol)
        for k in range(1):
            for i in range(self.ncol):
                zt[i] = self.__predict_sample(i=i, z=z, zt=zt)
        return(zt)
    
    def sample_knockoffs(self):
        Zt = np.zeros_like(self.Z)
        first = self.sample_row(self.Z[0, :])
        Zt[0, :] = first
        for i in range(1, self.Z.shape[0]):
            Zt[i, :] = self.sample_row(self.Z[i, :], Zt[i-1, :])
        return(Zt)
        