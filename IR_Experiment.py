#%% libreries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
def plot_DecisionSpace(model,X,t,e = 1,name = None):
        # X = model[1].transform(XX)
        nn = 100
        xmin,xmax = np.max(X[:,0]) + e , np.min(X[:,0]) - e
        ymin,ymax = np.max(X[:,1]) + e , np.min(X[:,1]) - e
        xx,yy = np.meshgrid(np.linspace(xmin,xmax,nn),np.linspace(ymin,ymax,nn))
        XX = np.vstack([xx.flatten(),yy.flatten()]).T
        TT = model.predict(XX)
        tt = TT.reshape(xx.shape)
        return xx,yy,tt
#%%

estimators = {'SVM':[('zscore',StandardScaler()),
                     ('clf',SVC())
                    ],
              'TWSVM':[('zscore',StandardScaler()),
                        ('clf',TWSVM())
                        ],
              'ETWSVM':[('zscore',StandardScaler()),
                        ('clf',ETWSVM())
                        ],
            #   'KSVM_S':[('zscore',StandardScaler()),
            #             ('clf',SVMK())
            #             ],
              'KSVM_TL1':[('zscore',StandardScaler()),
                        ('clf',SVMK())
                        ],
            #   'KTSVM_S':[('zscore',StandardScaler()),
            #             ('clf',TWSVM_krein())
            #             ],
              'KTSVM_TL1':[('zscore',StandardScaler()),
                        ('clf',TWSVM_krein())
                        ]
              }

C_list = [0.0001,0.001,0.01,0.1,1,10,100]

#%% datasets
Xs,ys = [],[]
for IR in [1,2,3,4,5]:
    noise = 0.1
    X,y  = make_moons(n_samples=400,noise=noise)
    y[y==0] = -1 

    ind_neg = np.where(y==-1)[0]
    ind_pos = np.where(y==1)[0]

    ind_pos = ind_pos[np.random.permutation(ind_pos.size)[0:int(ind_neg.size/IR)]]

    X= np.vstack([X[ind_neg],X[ind_pos]])
    y = np.hstack([y[ind_neg],y[ind_pos]])
    Xs.append(X)
    ys.append(y)

#%%
scores = {'acc': 'accuracy',
          'bal_acc':'balanced_accuracy',
          'f1_w':'f1_weighted'}
results = {}
for i,(X,y) in enumerate(zip(Xs,ys)):
     s0 = np.median(pdist(X))
     kernels_si = []
     gamma_list = [s for s in np.linspace(.1*s0,1.2*s0,5)]

     gamma_list = [2**(s) for s in range(-6,7)]
     gamma_list_twsvm = [s for s in np.linspace(.1*s0,1.2*s0,5)]
     for s in list(itertools.product(gamma_list,np.logspace(-2,2,5))):
          kernels_si.append( tanh_kernel(gamma=s[0],coef0=s[1]) )
     kernels_tl1 = [TL1(p) for p in np.logspace(-2,2,5).tolist() + [0.7*X.shape[1]]] 

     params = {'SVM':{'clf__C':C_list,
                     'clf__gamma':gamma_list},
            'TWSVM':{'clf__c1':[0.0001,0.001,0.1,1,10],
                    'clf__c2':[0.0001,0.001,0.1,1,10],
                    'clf__scale': gamma_list_twsvm},
            'ETWSVM':{'clf__c1':[0.0001,0.001,0.1,1,10],
                      'clf__c2':[0.0001,0.001,0.1,1,10],
                      'clf__kernfunction':['rbf'],
                      'clf__kernparam': gamma_list
                    },
            # 'KSVM_S':{'clf__C': C_list,
            #           'clf__kernel': kernels_si
            #             },
            'KSVM_TL1':{'clf__C': C_list,
                      'clf__kernel': kernels_tl1
                        },
            # 'KTSVM_S':{'clf__c1': C_list,
            #              'clf__c2':C_list,
            #              'clf__kernel': kernels_si
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
           cv = StratifiedKFold(2)
           grid = GridSearchCV(Pipeline(estimator),
                               param_grid,
                               cv=cv,
                               n_jobs=5,
                               scoring=scores,
                               refit='f1_w',
                               error_score='raise',
                               verbose=1
                               )
           grid.fit(X,y)

           best_model = grid.best_estimator_
           xx,yy,tt = plot_DecisionSpace(best_model,X,y)
           res_aux[model_name] = (xx,yy,tt)
     results[f"IR{i+1}"] = res_aux
     
# %%
joblib.dump(results,'results_IR_experiment.joblib')
#%%
results_dic = joblib.load('results_IR_experiment_v1.joblib')
#%%
colores_nombre = [
    'red',    # Rojo
    'blue',   # Azul
    'green',  # Verde
    'black', # negro
    'purple', # Morado
    'orange', # Naranja
    'cyan'    # Cian/Turquesa
]
patrones = [
     ':',
     '-',
     '--',
     '-.',
     'solid',
     'densely dashdotted',
     'densely dashed'

]
plt.rcParams.update({'font.size': 15})
fig,axes = (plt
            .subplots(1,5,figsize=(21,8))
            )
for ii,ax in enumerate(axes):
    ir = f'IR{ii+1}'
    X,y = Xs[ii],ys[ii]
    ax.scatter(X[:,0],
                X[:,1],
                c = y
                )
    legend_handles = []
    for i,(model,aux) in enumerate(results_dic[ir].items()):
        xx,yy,tt = aux
        ax.contour(xx,
                    yy,
                    tt,
                    levels = [0],
                    vmin=-1,
                    vmax=1,
                    colors=colores_nombre[i],
                    alpha = 0.5,
                    linestyles = patrones[i]
                    )
        ax.set_xlabel(f"IR={ii+1}")
        (legend_handles
        .append(mlines.Line2D([],
                              [],
                              color=colores_nombre[i],
                              label=r"$%s$" % model.replace('TL1','{T1}'),
                              linestyle=patrones[i]))
        )
    ax.set_aspect('equal')
    ax.axis('off')
plt.legend(handles=legend_handles,loc='upper center', ncol=7, bbox_to_anchor=(-1.9, 1.15))
plt.tight_layout()

fig.savefig("imgs/IR_experimento.pdf",dpi = 300,  bbox_inches='tight')
# %%
