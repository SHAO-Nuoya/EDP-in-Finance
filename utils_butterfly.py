import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.sparse import eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

class butterfly:
    """Summary"""
    def __init__(self, I, N):
        self.T = 0.1 
        self.sigma_min = 0.15
        self.sigma_max = 0.25
        self.K1 = 90
        self.K2 = 110
        self.r = 0.1
        self.Smin = 0
        self.Smax = 200
        self.I = I
        self.N = N
        self.St, self.h, self.DSt = self.DSt()

        #self.h = (self.Smax - self.Smin)/(I+1)
        self.dt = self.T/N
        #self.St = self.Smin+self.h*np.arange(1,self.I+1).reshape(-1, 1)
        
    def DSt(self):
        k = self.I/60
        step_lengths = np.array([10, 5, 2, 1, 0.5, 1, 2, 5, 10])/k
        intervals = np.array([self.Smin, 40, 80, 88, 98, 102, 112, 120, 160, self.Smax])
        mesh = []
        for i in range(len(step_lengths)):
            tmp = np.arange(intervals[i], intervals[i+1], step_lengths[i])
            mesh.append(tmp)
        St = np.concatenate(mesh)
        h = St[1:] - St[:-1]
        return St.reshape(-1, 1), h, sparse(np.diag(St))
        
    
    def U0(self):
        x = self.St
        return np.maximum(x - self.K1, 0) - 2*np.maximum(x-0.5*self.K1-0.5*self.K2, 0) + np.maximum(x - self.K2, 0)
    
    def b_plus(self):
        b = np.diag(np.maximum(-self.r * self.St, 0).reshape(-1, ))
        return sparse(b)
    
    def b_minus(self):
        b = np.diag(np.maximum(self.r * self.St, 0).reshape(-1, ))
        return sparse(b)
     
    def D_plus(self):
        D = np.zeros((self.I, self.I))
        ah = -3/(2*self.h)
        bh = 2/self.h
        ch = -1/(2*self.h)
        
        D[self.I-2, self.I-2] = ah[-1]
        D[self.I-2, self.I-1] = bh[-1]  
        D[self.I-1, self.I-1] = ah[-1]
        
        for i in range(self.I-2):
            D[i,i] = ah[i]
            D[i, i+1] = bh[i] 
            D[i, i+2] = ch[i]
            
        return sparse(D)
    
    def D_minus(self):
        D = np.zeros((self.I, self.I))
        ah = 3/(2*self.h)
        bh = -2/self.h
        ch = 1/(2*self.h)
        
        D[0, 0] = ah[0] 
        D[1, 0] = bh[0]
        D[1, 1] = ah[0]
        
        for i in range(2, self.I):
            D[i,i] = ah[i-1]
            D[i, i-1] = bh[i-1] 
            D[i, i-2] = ch[i-1]
            
        return sparse(D)
    
    def A_minus(self):
        A_ = np.eye(self.I)
        for i in range(1, self.I-1):
            A_[i-1, i] = -1
        return sparse(A_)
        
    def D2(self):
        D2 = np.zeros((self.I, self.I))
        h = self.h
        
        D2[0, 0] = -2/h[0]**2
        D2[0, 1] = 1/h[0]**2
        D2[self.I-1, self.I-2] = 1/h[-1]**2
        D2[self.I-1, self.I-1] = -2/h[-1]**2
        for i in range(1, self.I-1):
            D2[i, i-1] = 1/h[i]**2
            D2[i,i] = -2/h[i]**2
            D2[i,i+1] = 1/h[i]**2
            
        return sparse(D2)
    
    
    
    def newton(self, x, B, C, c, eta=0.001, k_max=100):
        k = 0
        F_prime = np.eye(B.shape[0])
        err = np.inf
        while (err > eta and k < k_max):
            for i in range(B.shape[0]):
                if (B @ x)[i][0] <= (C @ x)[i][0]:
                    F_prime[i] = B.toarray()[i]
                else:
                    F_prime[i] = C.toarray()[i]

            x_new = x - np.linalg.inv(F_prime) @ np.maximum(B @ x -c, C @ x -c)
            err = np.linalg.norm(np.maximum(B @ x -c, C @ x -c), np.inf)
            x = x_new.copy()
            k += 1
        return x
    
    def solve(self, U0, schema="BDF2"):
        U = U0.copy()
        A_minus, D2 = self.A_minus(), self.D2()
        r, dt, DSt = self.r, self.dt, self.DSt
        DSt2 = DSt**2
        b_plus, b_minus = self.b_plus(), self.b_minus()
        D_plus, D_minus = self.D_plus(), self.D_minus()
        
        B = -(r*dt + 1) * eye(self.I) + 0.5*dt*self.sigma_min**2 * DSt2 @ D2 + r*dt*DSt @ A_minus 
        C = -(r*dt + 1) * eye(self.I) + 0.5*dt*self.sigma_max**2 * DSt2 @ D2 + r*dt*DSt @ A_minus
        c =  - U.copy()
        U = self.newton(U, B, C, c)
        
        if schema=="EI":
            for t in range(1, self.N):
                c =  -U.copy()
                U = self.newton(U, B, C, c)

        elif schema == "BDF2":
            U_pre = U0.copy()
            B = -(1.5/dt+r)*eye(self.I) + 0.5*self.sigma_max**2*DSt2@D2 - b_plus@D_minus + b_minus@D_plus
            C = -(1.5/dt+r)*eye(self.I) + 0.5*self.sigma_min**2*DSt2@D2 - b_plus@D_minus + b_minus@D_plus
            
            for t in range(1,self.N):
                c = (-2*U + 0.5*U_pre)/dt
                U_pre = U.copy()
                U = self.newton(U, B, C, c)
            
        return U

def schema_plot_butterfly(I, N, schema="EI"):
    UV = butterfly(I=I, N=N)
    U0 = UV.U0()
    U = UV.solve(U0, schema=schema)
    St = UV.St
 
    plt.plot(St, U0, label='U0')
    plt.plot(St, U, label=schema)
    plt.title('I={}, N={}'.format(I,N))

    plt.legend()
    plt.grid()
    
