#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 17:53:56 2019
@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:51:19 2019
@author: felipe
"""

#from sklearn.datasets import load_wine
#from scipy.stats import zscore
#
#import time
#from scipy.io import loadmat
#from sklearn.naive_bayes import GaussianNB

import numpy as np
#from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import  fmin

from sklearn.base import  BaseEstimator, TransformerMixin

#del inputMat
#%%

class RAdam(BaseEstimator, TransformerMixin):
    def __init__(self,
                 learning_rate     = 0.01,
                 min_lr            =  0.00001, 
                 beta_1            = 0.9,
                 beta_2            = 0.999,
                 epsilon           = 1e-7):

        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._min_lr = min_lr
        self.iteration = 0
        self.m = None
        self.v = None
    
    
    def step(self,gradient,theta):
        if self.iteration == 0:
            t = self.iteration + 1
            self.iteration = t
            m = np.zeros(theta.shape)
            self.m = m
            v = np.zeros(theta.shape)
            self.v = v
        else:
            t = self.iteration + 1

            
        beta1_t = np.power(self._beta1,t)
        beta2_t = np.power(self._beta2,t)
        
        sma_inf = 2.0/(1.0 - self._beta2) - 1.0
        sma_t = sma_inf - 2.0*t*beta2_t/(1.0 - beta2_t)
        
        self.m  = (self._beta1*self.m) + (1.0 - self._beta1)*gradient
        self.v  = (self._beta2*self.v) + (1.0 - self._beta2)*np.power(gradient,2)
        m_corr_t = self.m/(1.0 - beta1_t)
        
        if sma_t > 4:
            v_corr_t = np.sqrt(self.v/(1.0 - beta2_t))

            r_t = np.sqrt((sma_t - 4.0)/(sma_inf - 4.0) *
                          (sma_t - 2.0)/(sma_inf - 2.0) *
                           sma_inf/sma_t)
            theta_t = theta-self._lr*r_t*m_corr_t/(v_corr_t + self._epsilon)
        else:
            theta_t = theta-self._lr*m_corr_t
        
        return theta_t
#%% 
class Adam(BaseEstimator, TransformerMixin):
    def __init__(self,
                 learning_rate     = 0.01,
                 min_lr            =  0.00001, 
                 beta_1            = 0.9,
                 beta_2            = 0.999,
                 epsilon           = 1e-7):

        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._min_lr = min_lr
        self.iteration = 0
        self.m = None
        self.v = None
    
    
    def step(self,gradient,theta):
        if self.iteration == 0:
            self.iteration = self.iteration+1
            m = np.zeros(theta.shape)
            self.m = m
            v = np.zeros(theta.shape)
            self.v = v
        
        m = self._beta1*self.m + (1-self._beta1)*gradient
        v = self._beta2*self.v + (1-self._beta2)*np.power(gradient,2)
            
        m_hat = m/(1 - np.power(self._beta1,self.iteration))
        v_hat = v/(1 - np.power(self._beta2,self.iteration))
        
        theta = theta - self._lr*m_hat/(np.sqrt(v_hat)+self._epsilon)
        
        self.m = m
        self.v = v
            
        self.iteration = self.iteration+1
        return theta
    
#%% 
class NAdam(BaseEstimator, TransformerMixin):
    def __init__(self,
                 learning_rate     = 0.01,
                 min_lr            =  0.00001, 
                 beta_1            = 0.9,
                 beta_2            = 0.999,
                 epsilon           = 1e-7):

        self._lr = learning_rate
        self._beta1 = beta_1
        self._beta2 = beta_2
        self._epsilon = epsilon
        self._min_lr = min_lr
        self.iteration = 0
        self.m = None
        self.v = None
    
    
    def step(self,gradient,theta):
        if self.iteration == 0:
            self.iteration = self.iteration+1
            m = np.zeros(theta.shape)
            self.m = m
            v = np.zeros(theta.shape)
            self.v = v
        
        m = self._beta1*self.m + (1-self._beta1)*gradient
        v = self._beta2*self.v + (1-self._beta2)*np.power(gradient,2)
            
        m_hat = m/(1 - np.power(self._beta1,self.iteration))+(1-self._beta1)*gradient/(1-np.power(self._beta1,self.iteration))
        v_hat = v/(1 - np.power(self._beta2,self.iteration))
        
        theta = theta - self._lr*m_hat/(np.sqrt(v_hat)+self._epsilon)
        
        self.m = m
        self.v = v
            
        self.iteration = self.iteration+1
        return theta
            
#%%
class CKA_Adam(BaseEstimator, TransformerMixin):
        
    def __init__(self, showWindow = 1,showCommandLine = 1, lr = 1e-3,
                 min_grad = 1e-5, goal = -np.inf, epoch = 'default',
                 valid = 'defaulf',training = 'default', batch = 'default',
                 Q = 2, init = 'pca', max_fail = 10):
        self.showWindow = showWindow
        self.showCommandLine = showCommandLine
        self.lr = lr
        self.min_grad = min_grad
        self.goal = goal
        self.epoch = epoch
        self.batch = batch
        self.training = training
        self.valid = valid
        self.Q = Q
        self.init = init
        self.max_fail = max_fail
        
    def _kScaleOptimization(self,x,y=0,s0=0,obj_func='info',param=0):
        if y == 0:
            x = cdist(x,x)
        else:
            x = cdist(x,y)
        if s0 == 0:
            s0 = np.mean(x)
        if param == 0:
            alpha = 2
        else:
            alpha = param

        def info_obj_fun(s):
            k  = np.e**(-x**2/(2*s**2))
            vi = np.mean(k,0)**(alpha-1)
            return -np.var(vi)

        def var_obj_fun(s):
            k = np.e**(-x**2/(2*s**2))
            return -np.var(k)

        func = obj_func.lower()
        if func == 'info':
            #sopt = minimize(info_obj_fun,s0,method='Powell')
            sopt = fmin(info_obj_fun,s0,disp=0)
        elif func == 'var':
            #sopt = minimize(var_obj_fun,s0,method='Powell')
            sopt = fmin(var_obj_fun,s0,disp=0)
        else:
            print('unknown cost function')
        return sopt
    
    def _A_pca(self,X, d, sw=0):
        siz = np.array(X.shape)
        Xpp = np.matrix(X)
        if siz[0] > siz[1]:
            P = Xpp.T*Xpp # outter product pxp
        else:
            P = Xpp*Xpp.T # inner product nxn
            
        Val, Vec = np.linalg.eig(P)
        Val      = np.abs(Val)/Val.sum()
        # organizar de mayor a menor
        index = np.argsort(Val)# :
        index = index[::-1] # ordenar los datos de mayor a menor
        Val = Val[index]
        Vec = Vec[:,index]
        if d == 0: # valores propios mayores a la media de la suma de los valores propios
            index = Val >= np.mean(Val)
            W = Vec[:,index]
        elif d > 0 and d < 1:
            va = Val[0]
            W = Vec[:,0]
            i = 1
            while va < d:
                W  = np.concatenate((W,Vec[:,i]),1)
                va = va+Val[i]
                i += 1
        elif d >= 1:
            W = Vec[:,0:d]
        if siz[0] < siz[1]:
            Val1 = np.matrix(np.diag(Val[0:np.size(W,1)]**-0.5))
            W    = Xpp.T*W*Val1
        if sw == 1:
            vals = np.matrix(np.diag(Val))
            Ro   = vals*np.abs(W)
            Ro   = Ro.sum(1)
            Ro   = np.array(Ro)
            plt.stem(Ro[:,0]);plt.show()
        return Xpp*W, W, Val, Vec

    def _A_derivativeAs(self,vecas,x,n,l,h):
        a  = np.real(np.reshape(vecas[0:-1],[np.shape(x)[1],-1],order='F'))
        u  = vecas[-1]
        s  = 10**u
        sp = self._kScaleOptimization(np.matmul(x,a))
        a  = a/(np.sqrt(2)*sp)
        y  = np.matmul(x,a)
        d  = cdist(y,y)
        k  = np.e**(-d**2/2)
        if np.any(np.isnan(np.ravel(k))):
            #print('whoa')
            f     = np.nan
            gradf = np.zeros(np.shape(vecas))
            rho   = np.nan
        else:
            trkl = np.trace(np.matrix(k)*np.matrix(h)*np.matrix(l)*np.matrix(h))
            trkk = np.trace(np.matrix(k)*np.matrix(h)*np.matrix(k)*np.matrix(h))

            grad_lk = np.matrix(h)*np.matrix(l)*np.matrix(h)/trkl
            grad_k  = 2*np.matrix(h)*np.matrix(k)*np.matrix(h)/trkk
            grad    = grad_lk - 0.5*grad_k
            p       = np.array(grad)*k

            p = (p+p.T)/2

            grada = np.matrix(x).T*(p-np.diagflat(np.matrix(p)*np.ones([n,1])))*(np.matrix(x)*a)
            grada = -4*np.real(np.ravel(grada,order='F'))
            grads = np.trace((-k*d**2)*np.real(grad))
            grads = s*np.log(10)*grads
            gradf = np.real(np.append(grada,grads))
            f     = -np.real(np.log(trkl)-np.log(trkk)/2)
            rho   = trkl/np.sqrt(trkk)
        return f, gradf, k, y, rho

    
    def _kITLMetricLearningMahalanobis(self,X,L,labels):
        N = np.shape(X)[0]
        Q = self.Q

        if self.init == 'pca':
            A_i = self._A_pca(X, Q)[1]
        else:
            A_i = self.init
            
        if isinstance(self.training, np.ndarray):
            #trInd = np.array(self.training,dtype=np.int)
            trInd = self.training 
        else:
            if self.training == 'default':
                trInd = np.ones([N],dtype=bool)
            
        if isinstance(self.valid, np.ndarray):
            valInd = self.valid
        else:
            if self.valid == 'defaulf':
                valInd = np.squeeze(np.logical_not(trInd)) 
        
        if self.epoch == 'default':
            epoch = 10
        else:
            epoch = self.epoch

        if self.batch == 'default':
            if 'labels' in vars():
                batch = 10
            else:
                batch = int(len(labels)/10)
        else:
            batch = self.batch

        #etav     = self.lr
        print_it = self.showCommandLine
        #plot_it  = self.showWindow
        #min_grad = self.min_grad
        goal     = self.goal
        max_fail = self.max_fail
        
        # initialization
        Xval       = X[valInd,:]
       	X          = X[trInd,:]
       	Lval       = L[valInd,:][:,valInd]
       	L          = L[trInd,:][:,trInd]
#       	labels_val = labels[valInd]
       	labels     = labels[trInd]
       	Nval       = np.sum(valInd)
       	N          = np.sum(trInd)
        
       	#for training data
       	sopt = self._kScaleOptimization(np.matmul(X,A_i))
       	A    = A_i/(np.sqrt(2)*sopt)
       	s0   = 1/(2*sopt**2)
       	u_i  = np.log(s0)/np.log(10)
       
       	vecAs   = np.hstack([np.ravel(A,order='F'),u_i])#np.vstack([np.reshape(A,[np.size(A),1]),u_i])
       	#H       = np.eye(N) - 1.0/N*np.ones([N,1])*np.ones([1,N])
        H = np.eye(batch) - 1.0/batch*np.ones([batch,1])*np.ones([1,batch])  
       	if Nval == 0:
       		Hval = np.array([])
       	else:
       		Hval = np.eye(Nval)-1.0/Nval*np.ones([Nval,1])*np.ones([1,Nval])
       	eta = self.lr
        #eta_start = etav[0]
       	#eta_end   = etav[1]
       	F         = np.zeros([epoch,2])
       	fnew      = np.inf
       	fbest     = np.inf
       	checks    = 0
       	exitflag  = 3
        

        #%% Optimizacion 
        model_RAdam = RAdam(learning_rate=self.lr)
        
        for ii in np.arange(1,epoch):
            #print('Epoch %d of %d\n' % (ii,epoch))
       	    fold    = fnew
            indices = np.random.permutation(len(labels))
            X_epoch = X[indices,:]
            labels  = labels[indices]
            L_epoch = L[indices,:][:,indices]

            N_aux = batch
            for jj in range(0,len(labels)-batch,batch):
                fnew,gradf,K,Y,rho = self._A_derivativeAs(vecAs,X_epoch[jj:jj+batch],N_aux,
                                                       L_epoch[jj:jj+batch,jj:jj+batch],H)              
                F[ii,0] = fnew
                if Nval > 0:
                    F[ii,1],aa,K,Y,rho = self._A_derivativeAs(vecAs,Xval,Nval,Lval,Hval)
#                    labels = labels_val
                    if F[ii,1] > fbest:
                        checks = 0
                        fbest = F[ii,1]
                    if fnew > fold:
                        eta = eta  - eta * 0.1
                        self.lr = eta
           			#eta_start = eta_start - eta_start * 0.1
           			 #eta_end = eta_end - eta_end * 0.25
#           		if ii < epoch/2:
#           			eta = eta_start
#           		else:
#           			eta = eta_end
                if np.abs(np.linalg.norm(gradf)-np.linalg.norm(np.real(gradf))) > 0:
                    exitflag= -1
                    if print_it == 1:
                        print('Negative eigenvalues found.\n')
                        break
                dg = np.linalg.norm(gradf)                   
           		#vecAs = vecAs - eta * gradf
			# Calcular actualalizacion
                vecAs = model_RAdam.step(gradf,vecAs)
                
                if print_it & (np.mod(jj,10) == 0 or jj == 1):
           			#print('%d-%d -- eta = %.2e -- f = %.2e -- |df_dx| = %.2e\n' % (ii,epoch,eta,fnew,dg))
                     print('%d-%d -- eta = %.2e -- f = %.2e -- |df_dx| = %.2e\n' % (ii,epoch,0,fnew,dg))
           

                if checks >= max_fail:
                    exitflag = 1
                    if print_it:
                        print('Metric Learning done... Fails = %d \n' %(checks))
                    break
                if fnew < goal:
                    exitflag = 2
                    if print_it:
                        print('Metric Learning done...Goal = %f \n' %(fnew))
                    break
       	A  = np.real(np.reshape(vecAs[0:-1],[np.shape(X)[1],-1],order='F'))                
        sp = self._kScaleOptimization(np.dot(X,A))
       	A  =  A/(np.sqrt(2)*sp)
       	s  = vecAs[-1]
       	s  = 10**s
       	F  = F[0:ii,:]

       	return A, s, K, F, exitflag
       	    
                                   
    def fit(self,X,y, *_): 
        # X[muestras x features], datos de entrada
        # y[labels x 1], vectro de etiquetas
        y = np.squeeze(y)
        idxs = np.argsort(y,kind='stable')
        y = y[idxs]
        X = X[idxs,:]
        KL = np.asmatrix(y).T == np.asmatrix(y)
        
        if isinstance(self.training, np.ndarray):
            self.training = self.training[idxs]
        if isinstance(self.valid, np.ndarray):
            self.valid = self.valid[idxs]
        
        self.KL = KL
        self.Wcka, self.s,self.K,self.F,self.exisfalg = self._kITLMetricLearningMahalanobis(X,KL,y)
        return self 

    def transform(self, Xraw, *_):
        return  np.matrix(Xraw)*np.matrix(self.Wcka)

