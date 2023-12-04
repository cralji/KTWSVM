#%% libreries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.manifold import TSNE
from SVM_Krein.estimators import SVMK
from estimators import TWSVM
from sklearn.svm import SVC
from TWSVM_Krein.estimators import TWSVM_krein
from SVM_Krein.kernels import tanh_kernel,TL1
from estimators import ETWSVM
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,StratifiedKFold
from sklearn.pipeline import Pipeline
import joblib

import itertools

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.spatial.distance import pdist 

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import matplotlib

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from umap import UMAP

def kreinT1(XA,XB,gamma=0.1):
  DAB = cdist(XA,XB,metric='cityblock')
  KAB = np.maximum(gamma - DAB,np.zeros(DAB.shape))
  return KAB

def Kcen(K):
  N = K.shape[0]
  H = np.eye(N) - (1/N)*np.ones((N,1)).dot(np.ones((1,N)))
  return H.dot(K).dot(H)


def K2D(K):
  D = np.zeros(K.shape)
  Nr,Nc = K.shape
  D = np.repeat(np.diag(K).reshape(1,-1),Nr,axis=0) - 2*K + np.repeat(np.diag(K).reshape(-1,1),Nc,axis=1)
  return D


import warnings
# Ignorar todas las advertencias
warnings.filterwarnings('ignore')




def plot_DecisionSpace(model,X,t,e = 1,name = None):
        # X = model[1].transform(XX)
        nn = 100
        xmin,xmax = np.max(X[:,0]) + e , np.min(X[:,0]) - e
        ymin,ymax = np.max(X[:,1]) + e , np.min(X[:,1]) - e
        xx,yy = np.meshgrid(np.linspace(xmin,xmax,nn),np.linspace(ymin,ymax,nn))
        XX = np.vstack([xx.flatten(),yy.flatten()]).T
        TT = model.predict(XX)
        tt = TT.reshape(xx.shap
                        
                        e)
        return xx,yy,tt
#%%

estimators = {
            #   'SVM':[('zscore',StandardScaler()),
            #          ('clf',SVC())
            #         ],
            #   'TWSVM':[('zscore',StandardScaler()),
            #             ('clf',TWSVM())
            #             ],
            #   'ETWSVM':[('zscore',StandardScaler()),
            #             ('clf',ETWSVM())
            #             ],
            #   'KSVM_S':[('zscore',StandardScaler()),
            #             ('clf',SVMK())
            #             ],
            #   'KSVM_TL1':[('zscore',StandardScaler()),
            #             ('clf',SVMK())
            #             ],
               # 'KTSVM_S':[('zscore',StandardScaler()),
               #           ('clf',TWSVM_krein())
               #           ],
              'KTSVM_TL1':[('zscore',StandardScaler()),
                        ('clf',TWSVM_krein())
                        ]
              }

C_list = [0.001,0.01,0.1,1,10]

#%% datasets
path_data = './data'
paths_file = ['{}/{}'.format(path_data,_) for _ in os.listdir(path_data)]
paths_file.sort()

paths_file.remove('./data/page-blocks0.dat')
paths_file.remove('./data/segment0.dat')
paths_file.remove('./data/Cryotherapy.xlsx')
#%%
scores = {'acc': 'accuracy',
          'bal_acc':'balanced_accuracy',
          'f1_w':'f1_weighted'}
results = {}
for i,path_dataset in enumerate(paths_file[6:7]):
     data = pd.read_table(path_dataset,delimiter=',',header=None).to_numpy()
     # Data
     X,t = data[:,0:-1].astype(np.float32),data[:,-1].astype(np.str_)
    
     labels = np.unique(t)
    
     nC = labels.size
     n_per_class = [sum(t==labels[i]) for i in range(nC)]
     # who_is_minoritary = np.argmin(n_per_class)
     who_is_majority = np.argmax(n_per_class)
     y = np.ones_like(t)
     y[t==labels[who_is_majority]] = -1
     y = y.astype(np.int8)


     s0 = np.median(pdist(X))
     kernels_si = []
     gamma_list = [s for s in np.linspace(.1*s0,1.2*s0,5)]

     gamma_list = [2**(s) for s in range(-4,5)]
     gamma_list_twsvm = [s for s in np.linspace(.1*s0,1.2*s0,5)]
     for s in list(itertools.product(gamma_list,np.logspace(-2,2,5))):
          kernels_si.append( tanh_kernel(gamma=s[0],coef0=s[1]) )
     # kernels_tl1 = [TL1(p) for p in np.logspace(-2,2,5).tolist() + [0.7*X.shape[1]]]
     kernels_tl1 = [TL1(p) for p in  [0.7*X.shape[1]]]

     params = {
            # 'SVM':{'clf__C':C_list,
            #          'clf__gamma':gamma_list},
            # 'TWSVM':{'clf__c1':[0.0001,0.001,0.1,1,10],
            #         'clf__c2':[0.0001,0.001,0.1,1,10],
            #         'clf__scale': gamma_list_twsvm},
            # 'ETWSVM':{'clf__c1':[0.0001,0.001,0.1,1,10],
            #           'clf__c2':[0.0001,0.001,0.1,1,10],
            #           'clf__kernfunction':['rbf'],
            #           'clf__kernparam': gamma_list
            #         },
            # 'KSVM_S':{'clf__C': C_list,
            #           'clf__kernel': kernels_si
            #             },
            # 'KSVM_TL1':{'clf__C': C_list,
            #           'clf__kernel': kernels_tl1
            #             },
            # 'KTSVM_S':{'clf__c1': C_list,
            #               'clf__c2':C_list,
            #               'clf__kernel': kernels_si
            #             },
            'KTSVM_TL1':{'clf__c1': C_list,
                         'clf__c2':C_list,
                         'clf__kernel': kernels_tl1
                        }
            }
     res_aux = {}
     for model_name,estimator in estimators.items():
           param_grid = params[model_name]
        #    cv = StratifiedShuffleSplit(n_splits=5,
        #                                test_size=0.3,
        #                                random_state=19931218
        #                                )
           cv = StratifiedKFold(5)
           grid = GridSearchCV(Pipeline(estimator),
                               param_grid,
                               cv=cv,
                               n_jobs=3,
                               scoring=scores,
                               refit='f1_w',
                               error_score='raise',
                               verbose=1
                               )
           grid.fit(X,y)

           best_model = grid.best_estimator_
           print("************* trained model *************")
           for perplexity in [5,10]:
                 red = TSNE(n_components=2,
                            random_state=42,
                            perplexity = perplexity,
                            metric='precomputed',
                            init='random')
                 red_umap = UMAP(n_components = 2,
                                 metric = 'precomputed',
                                n_neighbors = round(0.5*np.sqrt(X.shape[0])),
                                min_dist =0.9,
                                random_state=42)
                 red_ = TSNE(n_components=2,
                            random_state=42,
                            perplexity = perplexity,
                            # metric='precomputed',
                            init='random')
                 red_umap_ = UMAP(n_components = 2,
                                #  metric = 'precomputed',
                                n_neighbors = round(0.5*np.sqrt(X.shape[0])),
                                min_dist =0.9,
                                random_state=42)

                 mmax_ = MinMaxScaler()
                 Xt = best_model[0].transform(X)
                 KAB = best_model[1].kernel(Xt,Xt)
                 KAB_ = Kcen(KAB)
                 DK = K2D(KAB_)
                 try:
                     s,U = np.linalg.eig(DK)
                     plt.figure()
                     plt.stem(s)
                      # plt.savefig(f"./imgs_proj/001_eigenvalues_{path_dataset.split('/')[-1].split('.')[0]}_{model_name}_pre_{perplexity}.pdf")
                     plt.show()
                     s,U = s.real,U.real
                     ind_s = np.where(np.abs(s)>1e-3)[0]
                     s,U = s[ind_s],U[:,ind_s]
                     ind_pos = np.where(s>0)[0]
                     ind_neg = np.where(s<0)[0]
                     if ind_neg.size != 0:
                         s_pos,U_pos = s[ind_pos],U[:,ind_pos]
                         s_neg,U_neg = s[ind_neg],U[:,ind_neg]
                         K_pos = U_pos@np.diag(s_pos)@U_pos.T
                         # K_pos_ = Kcen(K_pos)
                         K_pos = K2D(K_pos)
                         K_neg = -1*U_neg@np.diag(s_neg)@U_neg.T
                         # K_neg_ = Kcen(K_neg)
                         K_neg = K2D(K_neg)
                         inds = np.where(np.abs(K_neg)<1e-6)
                         K_neg[inds] = 0
                         inds = np.where(np.abs(K_pos)<1e-6)
                         K_pos[inds] = 0
                        #  try:
                         K1_tra_tsne = mmax_.fit_transform(red.fit_transform(K_pos))
                        #  except:
                        #     K1_tra_tsne = mmax_.fit_transform(red_.fit_transform(K_pos))
                        #  try:
                         K2_tra_tsne = mmax_.fit_transform(red.fit_transform(K_neg))
                        #  except:
                        #     K2_tra_tsne = mmax_.fit_transform(red_.fit_transform(K_neg))
                        #  try:
                         K1_tra_upme = mmax_.fit_transform(red_umap.fit_transform(K_pos))
                        #  except:
                        #     K1_tra_upme = mmax_.fit_transform(red_umap_.fit_transform(K_pos))
                        #  try:
                         K2_tra_upme = mmax_.fit_transform(red_umap.fit_transform(K_neg))
                        #  except:
                        #     K2_tra_upme = mmax_.fit_transform(red_umap_.fit_transform(K_neg))
                         K_tra_tsne = mmax_.fit_transform(red_.fit_transform(DK))
                         K_tra_upme = mmax_.fit_transform(red_umap_.fit_transform(DK))

                         
                         fig,axes = plt.subplots(1,3,figsize = (9,5))
                         plt.subplots_adjust(wspace=0.4, hspace=0.4)
                         axes[0].scatter(K_tra_tsne[:,0],K_tra_tsne[:,1],c=y)
                     #     axes[0].set_title(r"$K_1$")
                         axes[0].set_aspect("equal",adjustable='box')
                         axes[0].axis("off")
                         axes[1].scatter(K1_tra_tsne[:,0],K1_tra_tsne[:,1],c=y)
                     #     axes[1].set_title(r"$K_2$")
                         axes[1].set_aspect("equal",adjustable='box')
                         axes[1].axis("off")
                         axes[2].scatter(K2_tra_tsne[:,0],K2_tra_tsne[:,1],c=y)
                     #     axes[2].set_title(r"$\tilde{K}$")
                         axes[2].set_aspect("equal",adjustable='box')
                         axes[2].axis("off")
                         fig.savefig(f"imgs_proj/tnse_{path_dataset.split('/')[-1].split('.')[0]}_{model_name}_pre_{perplexity}.pdf",
                                      dpi=400,
                                      bbox_inches='tight'
                                      )
                         fig.show()

                         fig,axes = plt.subplots(1,3,figsize = (9,5))
                         plt.subplots_adjust(wspace=0.4, hspace=0.4)
                         axes[0].scatter(K_tra_upme[:,0],K_tra_upme[:,1],c=y)
                     #     axes[0].set_title(r"$K_1$")
                         axes[0].set_aspect("equal",adjustable='box')
                         axes[0].axis("off")
                         axes[1].scatter(K1_tra_upme[:,0],K1_tra_upme[:,1],c=y)
                     #     axes[1].set_title(r"$K_2$")
                         axes[1].set_aspect("equal",adjustable='box')
                         axes[1].axis("off")
                         axes[2].scatter(K2_tra_upme[:,0],K2_tra_upme[:,1],c=y)
                     #     axes[2].set_title(r"$\tilde{K}$")
                         axes[2].set_aspect("equal",adjustable='box')
                         axes[2].axis("off")
                         fig.savefig(f"imgs_proj/upme_{path_dataset.split('/')[-1].split('.')[0]}_{model_name}_pre_{perplexity}.pdf",
                                      dpi=400,
                                      bbox_inches='tight'
                                      )
                         fig.show()
                 except Exception as e:
                     print(e)
                     pass
                     
                        
# %%
