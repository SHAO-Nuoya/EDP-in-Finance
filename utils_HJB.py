import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.sparse import csr_matrix as sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm

class Eikonal:
    def __init__(self, I, N, T, Smin = -3, Smax = 3):
        self.c = 1
        self.T = T
        self.Smin = Smin
        self.Smax = Smax
        self.I = I
        self.N = N
        
        self.h = (self.Smax - self.Smin)/(I+1)
        self.dt = self.T/N
    
        self.St = self.Smin+self.h*np.arange(1,self.I+1).reshape(-1, 1)
        
    
    # Dirichlet condition
    def vl(self):
        return 0
    
    
    def vr(self):
        return 0
    
    
#     def U0(self):
#         return -np.maximum(1-self.St**2,0)**2
    
    
    def D_minus(self):
        D = np.zeros((self.I, self.I))
        D[0, 0] = self.c/self.h
        
        for i in range(1, self.I):
            D[i,i] = self.c/self.h
            D[i, i-1] = -self.c/self.h
            
        return sparse(D)


    def q_minus(self):
        q = np.zeros((self.I,1))
        q[0] = -self.vl()*self.c/self.h
        return sparse(q)
    
    
    def D_plus(self):
        D = np.zeros((self.I, self.I))
        D[self.I-1, self.I-1] = -self.c/self.h
        
        for i in range(self.I - 1):
            D[i,i] = -self.c/self.h
            D[i, i+1] = self.c/self.h
            
        return sparse(D)
    
    
    def q_plus(self):
        q = np.zeros((self.I,1))
        q[0] = self.vr()*self.c/self.h
        return sparse(q)
    
    def D_minus_1(self):
        D = np.zeros((self.I, self.I))
        ah = 3/(2*self.h)
        bh = -2/self.h
        ch = 1/(2*self.h)
        
        D[0, 0] = ah 
        D[1, 0] = bh
        D[1, 1] = ah
        
        for i in range(2, self.I):
            D[i,i] = ah
            D[i, i-1] = bh 
            D[i, i-2] = ch
            
        return sparse(D)
    
    def D_plus_1(self):
        D = np.zeros((self.I, self.I))
        ah = -3/(2*self.h)
        bh = 2/self.h
        ch = -1/(2*self.h)
        
        D[self.I-2, self.I-2] = ah
        D[self.I-2, self.I-1] = bh  
        D[self.I-1, self.I-1] = ah
        
        for i in range(self.I-2):
            D[i,i] = ah
            D[i, i+1] = bh 
            D[i, i+2] = ch
            
        return sparse(D)
    
    
    
    def solve(self, U0):
        U = U0.copy()
        D_minus = self.D_minus()
        q_minus = self.q_minus()
        D_plus = self.D_plus()
        q_plus = self.q_plus()

        for n in range(1, self.N):
            t = n*self.dt
            U = U - self.dt*np.maximum(D_minus@U+q_minus, -D_plus@U-q_plus)

            
        return U
    
    def solve_1(self, U0):
        U = U0.copy()
        D_minus = self.D_minus_1()
        D_plus = self.D_plus_1()

        for n in range(1, self.N):
            U = U - self.dt*np.maximum(self.c*D_minus@U, -self.c*D_plus@U)
        return U
    
    def RK2_solve(self, U0):
        U = U0.copy()
        D_minus = self.D_minus_1()
        D_plus = self.D_plus_1()

        for n in range(1, self.N):
            U1 = U - self.dt*np.maximum(self.c*D_minus@U, -self.c*D_plus@U)
            U2 = U1 - self.dt*np.maximum(self.c*D_minus@U1, -self.c*D_plus@U1)
            U = (U+U2)/2
        return U
    
    def point_S(self, U, S):
        i = 0
        while self.St[i] < S:
            i += 1
        if self.St[i] == S:
            U_s = U[i]
        else:
            U_s = ((self.St[i] - S) * U[i-1] + (S - self.St[i - 1])*U[i])/self.h
        return U_s 
    

    
def v0(x):
    return -np.maximum(1 - x**2,0)**2
    
def exact(x, t = 1, c = 1):
    r1 = v0(x + c * t)
    r2 = v0(x - c * t)
    if x + c * t < 0:
        return r1
    if x - c * t >0:
        return r2
    
    return v0(0)

def exact_bis(x, t=0.4, c=1):
    r1 = -v0(x-c*t)
    r2 = -v0(x+c*t)
    if x+c*t < 0:
        return r1
    if x-c*t >0:
        return r2
    return np.minimum(r1, r2)

def schema_plot(I, N, T=1, neg=False, q121 = False, schema='0'):
    
    if q121==True:
        EI = Eikonal_1_2(I=I, N=N,T=T)
        
    else:
        EI = Eikonal(I=I, N=N,T=T)
    St = EI.St
    if neg==True:
        U0 = -v0(St)
        U = EI.solve(U0)
        U_exact = np.array([exact_bis(st) for st in St], dtype=object)
    else:
        U0 = v0(St)
        if schema == '0':
            U = EI.solve(U0)
        elif schema == '1':
            U = EI.solve_1(U0)
        elif schema == 'RK2':
            U = EI.RK2_solve(U0)
            
        U_exact = np.array([exact(st) for st in St], dtype=object)
       
        
    plt.plot(St, U0, color = 'g',label = 'U0')
    plt.plot(St, U,'--', color = 'b', label = 'EE schema')
    plt.plot(St, U_exact, color='r', label = 'exact')
    plt.title('EE, I={}, N={}'.format(I,N))
    plt.ylim(-1.4, 1.4)
    plt.xlabel('x')
    plt.ylabel('prix')
    plt.legend()
    plt.grid()

def schema_plot_IE(I, N, T=1, neg=False, q121 = False, schema='newton'):
    
    EI = Eikonal_IE(I=I, N=N,T=T)
    St = EI.St
    U0 = v0(St)
    U = EI.solve(U0, schema=schema)
    U_exact = np.array([exact(st) for st in St], dtype=object)
        
    plt.plot(St, U0, color = 'g',label = 'U0')
    plt.plot(St, U,'--', color = 'b', label = 'IE schema')
    plt.plot(St, U_exact, color='r', label = 'exact')
    plt.title('EE, I={}, N={}'.format(I,N))
    plt.ylim(-1.4, 1.4)
    plt.xlabel('x')
    plt.ylabel('prix')
    plt.legend()
    plt.grid()

    
class Eikonal_IE:
    def __init__(self, I, N, T, Smin = -3, Smax = 3):
        self.c = 1
        self.T = T
        self.Smin = -3
        self.Smax = 3
        self.I = I
        self.N = N
        
        self.h = (self.Smax - self.Smin)/(I+1)
        self.dt = self.T/N
    
        self.St = self.Smin+self.h*np.arange(1,self.I+1).reshape(-1, 1)
    
    
    def U0(self):
        return -np.maximum(1-self.St**2,0)**2
    
    
    def A_minus(self):
        A = np.zeros((self.I, self.I))
        
        A[0, 0] = 1
        for i in range(1, self.I):
            A[i,i] = 1
            A[i, i-1] = -1
            
        return sparse(A/self.h)
    
    def A_plus(self):
        A = np.zeros((self.I, self.I))
        
        A[self.I-1, self.I-1] = 1
        for i in range(self.I-1):
            A[i,i] = 1
            A[i, i+1] = -1
            
        return sparse(A/self.h)
    
    def q_minus(self):
        q = np.zeros((self.I,1))
        return sparse(q)
    
    def q_plus(self):
        q = np.zeros((self.I,1))
        return sparse(q)
    
    def D_minus(self):
        D = np.zeros((self.I, self.I))
        ah = 3/(2*self.h)
        bh = -2/self.h
        ch = 1/(2*self.h)
        
        D[0, 0] = ah 
        D[1, 0] = bh
        D[1, 1] = ah
        
        for i in range(2, self.I):
            D[i,i] = ah
            D[i, i-1] = bh 
            D[i, i-2] = ch
            
        return sparse(D)
    
    def D_plus(self):
        D = np.zeros((self.I, self.I))
        ah = -3/(2*self.h)
        bh = 2/self.h
        ch = -1/(2*self.h)
        
        D[self.I-2, self.I-2] = ah
        D[self.I-2, self.I-1] = bh  
        D[self.I-1, self.I-1] = ah
        
        for i in range(self.I-2):
            D[i,i] = ah
            D[i, i+1] = bh 
            D[i, i+2] = ch
            
        return sparse(D)
    
    
    def newton(self, x, B, b, C, c, eta, k_max):
        k = 0
        err = 1
        errs = [err]
        F_prime = np.eye(B.shape[0])

        while (errs[-1] > eta and k < k_max):
            for i in range(B.shape[0]):
                if (B @ x - b)[i][0] >= (C @ x - c)[i][0]:
                    F_prime[i] = B[i]
                else:
                    F_prime[i] = C[i]

            x_new = x - np.linalg.inv(F_prime) @ np.maximum(B @ x - b, C @ x - c)
            err = np.linalg.norm(np.abs(x_new - x), np.inf)
            x = x_new.copy()
            k += 1
            errs.append(round(err, 5))
        return x, errs
    
        

    
    def solve(self, U0, schema="newton"):
        """
        shema: newton/second_order
        """
        U, dt, c, I = U0.copy(), self.dt, self.c, self.I

        # Use IE to get U1
        en = Eikonal_IE(I, self.N, self.T)
        A_plus = self.A_plus()
        A_minus = self.A_minus()
        q_plus = self.q_plus()
        q_minus = self.q_minus()
        
        B_plus = np.eye(self.I) + self.dt*A_plus
        B_minus = np.eye(self.I) + self.dt*A_minus
        
        if schema == "newton":
            for t in range(self.N):
                b_plus = U - self.dt * q_plus
                b_minus = U - self.dt * q_minus
                U, errs = self.newton(U, B_plus, b_plus, B_minus, b_minus, 0.0001, 100)
        elif schema == "second_order":
            U1, _ = en.newton(U, B_plus, U,B_minus ,U, 0.0001, 100)
            U_pre = U.copy()
            U = U1.copy()

            D_plus = self.D_plus()
            D_minus = self.D_minus()
            B_plus = 3*np.eye(I) - 2*dt*c*D_plus
            B_minus = 3*np.eye(I) + 2*dt*c*D_minus

            for t in range(1, self.N):
                b_plus = 4*U - U_pre
                b_minus = 4*U - U_pre
                U, errs = self.newton(U, B_plus, b_plus, B_minus, b_minus, 0.0001, 100)
        #                 print(f"Errors : \n", errs, '\n')
                U_pre = U.copy()

        return U
    
    def point_S(self, U, S):
        i = 0
        while self.St[i] < S:
            i += 1
        if self.St[i] == S:
            U_s = U[i]
        else:
            U_s = ((self.St[i] - S) * U[i-1] + (S - self.St[i - 1])*U[i])/self.h
        return U_s 
    

def compute_alpha(e0, e1, h0, h1):
    tmp1 = np.log(abs(e0)/abs(e1))
    tmp2 = np.log(h0/h1)
    return tmp1/tmp2


def evaluation(Is,Ns,T=1, S_max=3, S_min=-3, monotone=True, RK2=False):
    n = len(Is)
    us = np.zeros(n)
    for i in range(n):
            
        EI = Eikonal(N=Ns[i], I=Is[i], T=T, Smax=S_max, Smin=S_min)
        St = EI.St
        U0 = v0(St)
        if monotone == True:
            U = EI.solve(U0)
        elif RK2==True:
            U = EI.RK2_solve(U0)
        else:
            U = EI.solve_1(U0)
        US = EI.point_S(U, S=1.5)
        us[i] = np.round(US, 6)
        
    ek = np.round(us[1:]-us[:-1], 6)
    hs = (S_max-S_min)/(Is+1)
    h = hs[1:] - hs[:-1]
    alpha_k = np.zeros(n-2)
    for i in range(n-2):
        alpha_k[i] = np.round(compute_alpha(ek[i], ek[i+1], h[i], h[i+1]), 6)
    return us, ek, alpha_k

def evaluation_IE(Is,Ns,T=1, S_max=3, S_min=-3, schema="newton"):
    n = len(Is)
    us = np.zeros(n)
    for i in range(n):
            
        EI = Eikonal_IE(N=Ns[i], I=Is[i], T=T, Smax=S_max, Smin=S_min)
        St = EI.St
        U0 = v0(St)
        U = EI.solve(U0, schema=schema)
        US = EI.point_S(U, S=1.5)
        us[i] = np.round(US, 6)
        
    ek = np.round(us[1:]-us[:-1], 6)
    hs = (S_max-S_min)/(Is+1)
    h = hs[1:] - hs[:-1]
    alpha_k = np.zeros(n-2)
    for i in range(n-2):
        alpha_k[i] = np.round(compute_alpha(ek[i], ek[i+1], h[i], h[i+1]), 6)
    return us, ek, alpha_k

### create table for errors and their order
def table(Is, Ns, U_s, e_k, order_alpha_k, schema, caption="N=I"):
    df = pd.DataFrame(columns=['I','N','U(s)','e_k','order alpha_k'])
    df['I'] = Is
    df['N'] = Ns
    df['U(s)'] = np.round(U_s, 6)
    df['e_k'] = np.concatenate((['-'], np.round(e_k,6)))
    df['order alpha_k'] = np.concatenate((['-', '-'], np.round(order_alpha_k,6)))
    styles = [dict(selector="caption", props=[("text-align", "center"), ("font-size", "110%"), ("color", 'black')])] 
    caption = schema + ' ('+caption + ')' 
    df_styler = df.style.set_table_attributes("style='display:inline'" ).set_caption(caption).set_table_styles(styles)
    return df_styler

class Eikonal_1_2:
    def __init__(self, I, N, T=1):
        self.c = 1
        self.T = T
        self.Smin = -3
        self.Smax = 3
        self.I = I
        self.N = N
        
        self.h = (self.Smax - self.Smin)/(I+1)
        self.dt = self.T/N
    
        self.St = self.Smin+self.h*np.arange(1,self.I+1).reshape(-1, 1)
        
    
    # Dirichlet condition
    def vl(self):
        return 0
    
    
    def vr(self):
        return 0
    

    
    def D(self):
        D = np.zeros((self.I, self.I))
        D[0, 1] = self.c/(2*self.h)
        D[self.I-1, self.I-2] = -self.c/(2*self.h)
        
        for i in range(1, self.I-1):
            D[i,i-1] = -self.c/(2*self.h)
            D[i, i+1] = self.c/(2*self.h)
            
        return sparse(D)


    def q(self):
        q = np.zeros((self.I,1))
        q[0] = -self.vl()*self.c/(2*self.h)
        q[-1] = self.vr()*self.c/(2*self.h)
        return sparse(q)

    
    def solve(self, U0):
        U = U0.copy()
        D = self.D()
        q = self.q()

        for n in range(1, self.N):
            t = n*self.dt
            U = U - self.dt*np.maximum(D@U+q, -D@U-q)
            
        return U