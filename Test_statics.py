#%% Libreries
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
from SVM_Krein.kernels import tanh_kernel
from estimators import ETWSVM
import joblib

from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
#%% 
# models_name = ['ETWSVM_krein']
# folders_results = ['./results_RFF-TWSVM/']

models_name = ['SVM',
               'TWSVM',
               'ETWSVM',
               'KSVM_S',
               'KSVM_TL1',
               'KTWSVM_S',
               'KTWSVM_TL1']
folders_results = ['./results_SVM/',
                   './results_TWSVM/',
                   './results_ETWSVM/',
                   './results_KSVM/',
                   './results_KSVM_TL1/',
                   './results_KTWSVM/',
                   './results_KTWSVM_TL1/'
                   ]

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
Acc_all = []
GM_all = []
FM_all = []
for folder in tqdm(folders_results):
    mean = []
    std = []
    Acc = []
    GM = []
    FM = []
    for dataset in datasets:
        mdict = joblib.load(template.format(folder,dataset,5))
        for f in range(1,6):
          mdict_aux = mdict['Fold_{}'.format(f)]  
          Acc.append( mdict_aux['Acc'][0] )
          GM.append(mdict_aux['GM'][0])
          FM.append(mdict_aux['F1'][0])
    #     mean.append(np.mean(Acc))
    #     std.append(np.std(Acc))
    #     mean.append(np.mean(GM))
    #     std.append(np.std(GM))
    #     mean.append(np.mean(FM))
    #     std.append(np.std(FM))
    # colummns += [mean,std]
    Acc_all.append(Acc)
    GM_all.append(GM)
    FM_all.append(FM)
# %%
# %% Test Estadístico
metrics_values = {'Acc':Acc_all,
                  'GM':GM_all,
                  'FM':FM_all
                  }

models = [r'$SVM$',
          r'$TWSVM$',
          r'$ETWSVM$',
          r'$KSVM_S$',
          r'$KSVM_{T1}$',
          r'$KTSVM_S$',
          r'$KTSVM_{T1}$']
plt.rcParams.update({'font.size': 15})
fig,axes = (plt
            .subplots(1,
                      3,
                      figsize = (15,5)
                      )
            )
for i,(metric,values) in enumerate(metrics_values.items()):
   chi,pvalue = friedmanchisquare(*values)
   ph_friedman = (sp
                  .posthoc_nemenyi_friedman(np.array(values).T)
                  )
   ph_friedman.columns = models
   ph_friedman.index = models
   ph_friedman = ph_friedman[models[:-1]].iloc[1:]
   mask = np.triu(np.ones_like(ph_friedman, dtype=bool),k= 1)
#    mask[0][0] = False
#    mask[-1][-1] = False
   if i==0:
    (sns
        .heatmap(ph_friedman*100,
                annot = True,
                fmt = '.1f',
                vmin=0,
                vmax=100,
                square = True,
                cbar=False,
                mask = mask,
                cmap = sns.color_palette("viridis", as_cmap=True),
                ax = axes[i]
                )
    )
   else:
      (sns
       .heatmap(ph_friedman*100,
                annot = True,
                fmt = '.1f',
                vmin=0,
                vmax=100,
                square = True,
                cbar=False,
                yticklabels = False,
                mask = mask,
                cmap = sns.color_palette("viridis", as_cmap=True),
                ax = axes[i]
                )
        )
   axes[i].xaxis.tick_top()
   axes[i].set_title(r"%s | $\chi^2\!=\!%.2f$ p-value$\!=\!%.2e$" % (metric,chi,pvalue))
# Ajustar las margenes y reducir los tamaños entre los subplots
# plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1, hspace=0.1)

# Ajustar automáticamente los parámetros del subplot
plt.tight_layout()
fig.savefig("imgs/test_statics.pdf",  bbox_inches='tight',dpi =300)
# %%
