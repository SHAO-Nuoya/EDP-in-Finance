import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.sparse import csr_matrix as sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

class uncertain_vol:
    """_summary_"""
    def __init__(self, I, N, T, sigma):
        self.c = 1
        self.T = 0.5    
        self.Smin = -3
        self.Smax = 3
        self.I = I
        self.N = N
        self.sigma = sigma
        
        self.h = (self.Smax - self.Smin)/(I+1)
        self.dt = self.T/N
    
        self.St = self.Smin+self.h*np.arange(1,self.I+1).reshape(-1, 1)
        
    
    # Dirichlet condition
    def vl(self):
        return 1/2
    
    
    def vr(self):
        return -1/2
    
    
    def U0(self):
        x = self.St
        return np.sign(x)*(np.maximum(1-np.abs(x),0)**4-1)/2
    
    def q(self):
        q = np.zeros((self.I,1))
        q[0] = self.vl()/(self.h**2)
        q[-1] = self.vr()/(self.h**2)
        return sparse(q)
        
    def D2(self):
        D2 = np.zeros((self.I, self.I))
        h = self.h
        
        D2[0, 0] = -2/h**2
        D2[0, 1] = 1/h**2
        D2[self.I-1, self.I-2] = 1/h**2
        D2[self.I-1, self.I-1] = -2/h**2
        for i in range(1, self.I-1):
            D2[i, i-1] = 1/h**2
            D2[i,i] = -2/h**2
            D2[i,i+1] = 1/h**2
            
        return sparse(D2)
    
    
    def newton(self, x, B, b, c, eta, k_max) -> np.ndarray:
        k = 0
        err = 1
        errs = [err]
        F_prime = np.eye(B.shape[0])

        while (errs[-1] > eta and k < k_max):
            for i in range(B.shape[0]):
                if (B @ x - b)[i][0] <= (x - c)[i]:
                    F_prime[i] = B[i]

            x_new = x - np.linalg.inv(F_prime) @ np.minimum(B @ x - b, x - c)
            err = np.linalg.norm(np.abs(x_new - x), np.inf)
            x = x_new.copy()
            k += 1
            errs.append(round(err, 5))
        return x, errs
    
    def newton_bis(self, x, B, b, C, c, eta, k_max):
        k = 0
        err = 1
        errs = [err]
        F_prime = np.eye(B.shape[0])

        while (errs[-1] > eta and k < k_max):
            for i in range(B.shape[0]):
                if (B @ x - b)[i][0] <= (C @ x - c)[i][0]:
                    F_prime[i] = B[i]
                else:
                    F_prime[i] = C[i]

            x_new = x - np.linalg.inv(F_prime) @ np.maximum(B @ x - b, C @ x - c)
            err = np.linalg.norm(np.abs(x_new - x), np.inf)
            x = x_new.copy()
            k += 1
            errs.append(round(err, 5))
        return x, errs
    
    def solve(self, U0, schema="EE"):
        U = U0.copy()
        D2 = self.D2()
        q = self.q()
        if schema=="EE":
            for n in range(1, self.N):
                U -= self.dt*np.minimum(0, -self.sigma**2*(D2@U+q)/2)
        elif schema=="EI":
            
            B = np.eye(self.I)-self.dt*self.sigma**2*D2/2

            for t in range(self.N):
                b = U + self.dt*self.sigma**2*q/2
                c = U
                U, errs = self.newton(U, B, b, c, 0.0001, 100)
#             print(f"Errors : \n", errs, '\n')
        elif schema == "BDF2":
            uv_newton = uncertain_vol(self.I, self.N, self.T, self.sigma)
            B_IE = np.eye(self.I)-self.dt*self.sigma**2*D2/2
            b_IE = U + self.dt*self.sigma**2*q/2
            c_IE = U
            U1, _ = uv_newton.newton(U, B_IE, b_IE, c_IE, 0.0001, 100)

            U_pre = U.copy()
            U = U1.copy()

            B = 3*np.eye(self.I)
            C = B - self.dt*self.sigma**2*D2
            for t in range(self.N):
                b = 4*U - U_pre
                c = b + self.dt*self.sigma**2*q
                
                U_pre = U.copy()
                U, errs = self.newton_bis(U, B, b, C, c, 0.0001, 100)
#                 print(f"Errors : \n", errs, '\n')
        return U

def schema_plot(I, N, T=0.5, sigma=0.5, schema="EE"):
    UV = uncertain_vol(I=I, N=N, T=T, sigma=sigma)
    U0 = UV.U0()
    U = UV.solve(U0, schema=schema)
    St = UV.St
    plt.plot(St, U0, label='U0')
    plt.plot(St, U, label=schema)
    plt.title('I={}, N={}'.format(I,N))

    plt.legend()
    plt.grid()
    