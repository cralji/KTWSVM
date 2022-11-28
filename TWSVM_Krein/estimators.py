import numpy as np
# from TWSVM_Krein.utils import quadprog_solve_qp,Krein_EIG
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.gaussian_process.kernels import DotProduct
from TWSVM_Krein.solver import RPPHom,solver_minimize

class TWSVM_krein(BaseEstimator,ClassifierMixin):
    def __init__(self,
                 c1 = 0.001,
                 c2 = 1,
                 kernel = None,
                 epsilon = 1e-6):
        self.c1 = c1
        self.c2 = c2
        self.kernel = kernel
        self.epsilon = epsilon

    def fit(self,X,y):
        # self.y,self.labels = Targets2Labels(t)
        # y = self.y
        if self.kernel is None:
            self.kernel = DotProduct()
        indexPos = np.where(y == 1)[0]
        indexNeg = np.where(y == -1)[0]
        X_pos = X[indexPos][:]
        X_neg = X[indexNeg][:]
        N_pos = X_pos.shape[0]
        N_neg = X_neg.shape[0]

        # print(indexPos.shape,indexNeg.shape)
        sort_index = np.vstack([indexPos.reshape(-1,1),indexNeg.reshape(-1,1)]).reshape(-1,)
        # print(sort_index.shape)
        X = X[sort_index]
        y = y[sort_index]
        Q = X.shape[0]
        # print(X.shape,y.shape)
        # kern = kernel(kernfunction=self.kernel,kernparam=self.scale)
        
        u_pos = -1*np.ones((N_pos,1))
        u_neg = -1*np.ones((N_neg,1))

        PHI_pos = self.kernel(X_pos,X) # Npos x N
        PHI_neg = self.kernel(X_neg,X) # Nneg x N

        # print(PHI_pos.shape,u_pos.shape)
        S_pos = np.hstack([PHI_pos,u_pos]).T
        # print(S_pos.shape)
        S_neg = np.hstack([PHI_neg,u_neg]).T

        try:
            A_pos = np.linalg.inv(S_pos.dot(S_pos.T) +self.c1*np.eye(Q+1))
        except:
            A_pos = np.linalg.pinv(S_pos.dot(S_pos.T) +self.c1*np.eye(Q+1))
        try:
            A_neg = np.linalg.inv(S_neg.dot(S_neg.T) +self.c1*np.eye(Q+1))
        except:
            A_neg = np.linalg.pinv(S_neg.dot(S_neg.T) +self.c1*np.eye(Q+1))

        Hpos = S_neg.T.dot(A_pos.dot(S_neg))
        Hneg = S_pos.T.dot(A_neg.dot(S_pos))
        
        l_neg = np.zeros_like(u_neg).reshape(-1,)
        up_neg = self.c2*np.ones_like(u_neg).reshape(-1,)
        l_pos = np.zeros_like(u_pos).reshape(-1,)
        up_pos = self.c2*np.ones_like(u_pos).reshape(-1,)
        # RPPHom_ = RPPHom(max_iter=10000).reshape(-1,)
        
        alpha_neg = solver_minimize(Hpos+np.eye(Hpos.shape[0])*self.epsilon,u_neg.reshape(-1,),l_neg,up_neg)
        # alpha_neg = RPPHom_.call(Hpos+np.eye(Hpos.shape[0])*self.epsilon,u_neg,l_neg,up_neg)

        # alpha_neg = quadprog_solve_qp(H_pos+ np.eye(H_pos.shape[0])*self.epsilon,self.c2, N_neg)

        # alpha_neg = solverSVM(H_pos,c21,m=N_neg,y = -1.0)
        alpha_pos = solver_minimize(Hneg+np.eye(Hneg.shape[0])*self.epsilon,u_pos.reshape(-1,),l_pos,up_pos)
        # alpha_pos = RPPHom_.call(Hneg+np.eye(Hneg.shape[0])*self.epsilon,u_pos,l_pos,up_pos)
        

        z_pos = -1*A_pos.dot(S_neg.dot(alpha_neg))
        z_neg = 1*A_neg.dot(S_pos.dot(alpha_pos))

        self.wpos = z_pos[:-1]
        self.bpos = z_pos[-1]

        self.wneg = z_neg[:-1]
        self.bneg = z_neg[-1]
        K = self.kernel(X,X)
        self.wpos_norm = np.sqrt(self.wpos.T.dot(K.dot(self.wpos)))
        self.wneg_norm = np.sqrt(self.wneg.T.dot(K.dot(self.wneg)))
        self.X = X

        return self

    def predict(self,X):
        # labels = self.labels
        phi = self.kernel(X,self.X)   #rff.transform(X)
        d_pos = np.abs(phi.dot(self.wpos) + self.bpos)/self.wpos_norm
        d_neg = np.abs(phi.dot(self.wneg) + self.bneg)/self.wneg_norm

        # F = np.hstack([d_pos,d_neg])
        F = d_neg - d_pos
        # inds_min = np.argmin(F,axis=1)
        # # print(inds_min.shape)
        # t_est = labels[inds_min]
        y_est = np.sign(F)
        y_est[y_est==0] = 1
        y_est[np.isnan(y_est)] = 1
        return y_est