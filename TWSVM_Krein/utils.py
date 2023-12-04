#%% Libraries
import numpy as np
from numpy.random import choice
from numpy.linalg import qr,inv
import quadprog
#%% 
def quadprog_solve_qp(P,y,C,m=None): # , q, G=None, h=None, A=None, b=None):
    if m is None:
        m = P.shape[0]
    dtype = P.dtype
    q = -np.ones((m, ))
    # print('q_dtype: {}'.format(q.dtype))
    G = np.vstack((np.eye(m)*-1,np.eye(m)))
    h = np.hstack((np.zeros(m), np.ones(m) * C))

    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    A = y.astype(dtype).reshape(1,-1)
    b = 0.0
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    try:
        alpha = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0] 
    except:
        alpha = quadprog.solve_qp(np.eye(m,dtype=np.float64), qp_a, qp_C, qp_b, meq)[0] 
    return alpha

def Krein_EIG(K):
    d,U = np.linalg.eig(K)
    D = np.diag(d)
    S = np.sign(D)
    try:
        K_tilde = (U@S@D@np.linalg.inv(U + 1e-6)).real    
    except:
        K_tilde = (U@S@D@U.T).real
    K_tilde = K_tilde.astype(np.float64)
    return K_tilde,U,S,D

#%%
class SimpleSolver:
    """
    SimpleSolver for SVM-based methods
    Implementation based on:
        Vishwanathan, S. V. M., & Narasimha Murty, M. (n.d.). SSVM: a simple SVM algorithm. 
        Proceedings of the 2002 International Joint Conference on Neural Networks. IJCNN’02 (Cat. No.02CH37290). 
        doi:10.1109/ijcnn.2002.1007516 
        https://ieeexplore.ieee.org/abstract/document/1007516
    Implemented by: craljimenez@utp.edu.co

    Input Args:
        G: Quadratic matrix (n x n) from QD Problem from a SVM-based method
        C: regularizer parameter (float)
        y: Default None, it's labels array for the case of SVM-based methods.
        is_SVM: Bool, it specifics if the solver is or not a SVM-based method.
    
    Return Args:
        alpha: array (n,1) Lagrange's Multipliers
    """
    def __init__(self,G,C,y=None,is_SVM=False,tol = 1e-6,max_iter=1000):
        self.G = G
        self.C = C
        self.y = y
        self.is_SVM = is_SVM
        self.tol = tol
        self.max_iter = max_iter
        n = self.G.shape[0]
        if self.is_SVM:
            self.Iw = [choice(np.where(self.y == 1)[0],size=1,replace=False)[0],
                choice(np.where(self.y == -1)[0],size=1,replace=False)[0]]
            self.Io = [i for i in range(n) if i not in self.Iw]
            self.IC = []
        else:
            self.Iw = choice(range(n),size=2,replace=False).tolist()
            self.Io = [i for i in range(n) if i not in self.Iw]
            self.IC = []
        if type(C)==int or type(C)==float:
            self.Slacks = C*np.ones((n,))
        else:
            self.Slacks = C
        self.alphas = np.zeros_like(self.Slacks)

    def SolveLinearSystem(self):
        n = self.G.shape[0]
        Gw = self.G[self.Iw]
        nw = len(self.Iw)
        if len(self.IC) == 0:
            GC = self.G[self.IC] # Ic x N 
        else:
            GC = np.zeros((len(self.IC),n))
        Q,R = qr(Gw.dot(Gw.T))
        if self.is_SVM:
            raise ValueError('Not implemented yet')
        else:
            b = Gw.dot(np.ones((n,)) - GC.T.dot(self.Slacks[self.IC]))
        alphas_w = inv(R).dot(Q.T).dot(b) # search inverse from cholesky
        self.alphas[self.Iw] = alphas_w
        return None  

    def simple_solver(self,log = False):
        iter = 0
        error = 1
        while iter<self.max_iter and error > self.tol:
            alphas_old = self.alphas
            self.SolveLinearSystem()
            alphas = self.alphas[self.Iw]
            #%% Actiation Constraint
            ind_0 = np.where(alphas<0)[0]
            ind_C = np.where(alphas>self.Slacks)[0]
            if ind_0.size > 0:
                self.Io += [self.Iw[i] for i in ind_0]
                self.Iw = [inds for inds in self.Iw if inds not in self.Io]
            if ind_C.size > 0:
                self.IC += [self.Iw[i] for i in ind_C]
                self.Iw = [inds for inds in self.Iw if inds not in self.IC]
            #%% Relaxing Constraing in IC
            Gw = self.G[self.Iw] # Nw x n
            if len(self.Io)!=0:
                G0 = self.G[self.Io] # No x n
                Aux = G0.dot(Gw.T.dot(self.alphas[self.Iw])-np.ones((len(self.Io),)))
                indx = np.where(Aux<0)[0]
                self.Iw += [self.Io[ind] for ind in indx]
                self.Io = [inds for inds in self.Io if inds not in self.Iw]
            #%% Relaxing Constraing in IC
            if len(self.IC) != 0:
                GC = self.G[self.IC] # Nc x n
                Aux = GC.dot(Gw.T.dot(self.alphas[self.Iw])-np.ones((len(self.IC),)))
                indx = np.where(Aux>0)[0]
                self.Iw += [self.IC[ind] for ind in indx]
                self.IC = [inds for inds in self.IC if inds not in self.Iw]
            
            error = np.linalg.norm(self.alphas-alphas_old)
            if log:
                print('iter:{} \t error:{} \t nIw:{} \t nIo \t nIC'.format(iter,
                                                                           error,
                                                                           len(self.Iw),
                                                                           len(self.Io),
                                                                           len(self.IC)))
            
            






            
            
        
        return 0
# %%
