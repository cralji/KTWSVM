#%% Libraries
from cmath import tanh
import pandas as pd
import numpy as np
import os
from time import time

from TWSVM_Krein.estimators import TWSVM_krein

from TWSVM_Krein.kernels import tanh_kernel

from sklearn.base import BaseEstimator,TransformerMixin,ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,make_scorer,recall_score,f1_score,balanced_accuracy_score,roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold,GridSearchCV,StratifiedShuffleSplit,RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import pdist



import itertools
from tqdm import tqdm

from joblib import dump
import pickle

#%%

steps = [
         [('zscore',StandardScaler()),
          ('clf',TWSVM_krein())]
        ]


name_models = ['Krein_TWSVM']

#%% List Datasets

path_data = './data'
paths_file = ['{}/{}'.format(path_data,_) for _ in os.listdir(path_data)]
if not('results' in os.listdir()):
    os.mkdir('results')

paths_file.sort()

paths_file.remove('./data/page-blocks0.dat')
paths_file.remove('./data/segment0.dat')
paths_file.remove('./data/Cryotherapy.xlsx')


#%%
nf = 5
cv1 = StratifiedKFold(n_splits=nf)
cv2 = StratifiedKFold(n_splits=nf)

scores = {'acc': 'accuracy',
          'bal_acc':'balanced_accuracy'}  

#for path_dataset in paths_file:
# path_dataset = paths_file[1]


for path_dataset in paths_file:
    data = pd.read_table(path_dataset,delimiter=',',header=None).to_numpy()
    # Data
    X,t = data[:,0:-1].astype(np.float32),data[:,-1].astype(np.str)
    
    labels = np.unique(t)
    
    nC = labels.size
    n_per_class = [sum(t==labels[i]) for i in range(nC)]
    # who_is_minoritary = np.argmin(n_per_class)
    who_is_majority = np.argmax(n_per_class)
    y = np.ones_like(t)
    y[t==labels[who_is_majority]] = -1
    y = y.astype(np.int8)
    f=1
    # skf = StratifiedKFold(n_splits=10,shuffle=False)
    
    print('{:*^50}'.format(path_dataset))
    C_list = [0.001,0.01,0.1,1,10]
    results_dict = {}
    for train_index, test_index in cv1.split(X, y,y):
        Xtrain,t_train = X[train_index],y[train_index]
        Xtest,t_test = X[test_index],y[test_index]
        
        s0 = np.median(pdist(Xtrain))
        kernels = []
        gamma_list = [s for s in np.linspace(.1*s0,1.2*s0,5)]
        for s in list(itertools.product(gamma_list,np.logspace(-2,2,5))):
            kernels.append( tanh_kernel(gamma=s[0],coef0=s[1]) )
        params_grids = [
                        {'clf__c1': C_list,
                         'clf__c2':C_list,
                         'clf__kernel': kernels
                        }
                      ]
        #break
        sen = []
        spe = []
        acc = []
        gm = []
        f1 = []
        time_train = []
        T_est = []
        best_params = []
        list_results = []
        for step,params_grid,name_model in zip(steps,params_grids,name_models):
            tik = time()
            pipe = Pipeline(step,memory='pipe_data_pc')
            grid_search = GridSearchCV(pipe,
                                        params_grid,
                                        scoring=scores,
                                        cv=cv2,
                                        verbose=0,
                                        error_score='raise',
                                        refit ='bal_acc',
                                        n_jobs=3
                                       )
            tok = time()
            time_train.append(tok-tik)
            grid_search.fit(Xtrain,t_train)
            results = grid_search.cv_results_
            t_est = grid_search.best_estimator_.predict(Xtest)
            sen.append( recall_score(t_test,t_est) )
            spe.append( recall_score(t_test,t_est,pos_label=-1) )
            acc.append(accuracy_score(t_test,t_est))
            gm.append(balanced_accuracy_score(t_test,t_est))
            f1.append(f1_score(t_test,t_est,))
            T_est.append(t_est)
            best_params.append(grid_search.best_params_)
            list_results.append(results)
            print('data_ {} ----> acc: {} \t gm:{} \t f1:{}'.format(path_dataset,
                                                                    acc[-1],
                                                                    gm[-1],
                                                                    f1[-1]))
            model_path = './results/model_{}_{}_f{}.p'.format(name_model,path_dataset[7:-4],f)
            # model_path_tf = './results/model_{}_f{}.h5'.format(path_dataset[7:-4],f)
            # grid_search.best_estimator_[1].model.save(model_path_tf)
            #model_path = './results/model_{}_f{}.p'.format(path_dataset[7:-4],f)  #'model_sujeto_'+str(sbj)+'_cka_featuresCSP_BCI2a_acc.p'
            pickle.dump(grid_search.best_estimator_,open(model_path, 'wb'))
        results_dict['Fold_{}'.format(f)] = {
                                            'best_param':list_results,
                                            'cv_results':results,
                                            'Sen':sen,
                                            'Spe':spe,
                                            'Acc':acc,
                                            'GM':gm,
                                            'F1':f1,
                                            'time':time_train,
                                            'train':(Xtrain,t_train),
                                            'test':(Xtest,t_test),
                                            'T_pred':T_est
                                            }
            
        dump(results_dict,'./results/results_{}_f{}.joblib'.format(path_dataset[7:-4],f))   #'sujeto_'+str(sbj)+'_cka_featuresCSP_BCI2a_acc.joblib')
        f += 1




#%%
