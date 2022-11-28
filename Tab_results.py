#%% Libreries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from SVM_Krein.estimators import SVMK
from SVM_Krein.kernels import tanh_kernel

import joblib
#%% 
# models_name = ['ETWSVM_krein']
# folders_results = ['./results_RFF-TWSVM/']

models_name = ['TWSVMK']
folders_results = ['./results/']

datasets = os.listdir('./data')
datasets.sort()
datasets = [dataset.split('.')[0] for dataset in datasets if '.dat' in dataset]

#%%
datasets.remove('page-blocks0')
datasets.remove('segment0')
# datasets.remove('Cryotherapy')
#

template = '{}results_{}_f{}.joblib'
colummns = []
for folder in tqdm(folders_results):
    mean = []
    std = []
    for dataset in datasets:
        Acc = []
        GM = []
        FM = []
        mdict = joblib.load(template.format(folder,dataset,5))
        for f in range(1,6):
          mdict_aux = mdict['Fold_{}'.format(f)]  
          Acc.append( mdict_aux['Acc'][0] )
          GM.append(mdict_aux['GM'][0])
          FM.append(mdict_aux['F1'][0])
        mean.append(np.mean(Acc))
        std.append(np.std(Acc))
        mean.append(np.mean(GM))
        std.append(np.std(GM))
        mean.append(np.mean(FM))
        std.append(np.std(FM))
    colummns += [mean,std]

# %%
name_columns = []
for model in models_name:
    name_columns.append( model )
    name_columns.append( model )

indices = []
for dataset in datasets:
    indices.append( (dataset,'Acc'))
    indices.append( (dataset,'GM'))
    indices.append( (dataset,'FM'))
index = pd.MultiIndex.from_tuples(indices,names=['dataset','metric'])
X = np.vstack(colummns).T

df = pd.DataFrame(X,columns=name_columns,index=index)
print(df)

# %%
df.to_excel('./SVM_krein.xlsx')
# %%
