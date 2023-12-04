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

#%%
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
               'KTSVM_S',
               'KTSVM_TL1']
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
    name_columns.append( f"{model}_mean" )
    name_columns.append( f"{model}_std" )

indices = []
for dataset in datasets:
    indices.append( (dataset,'Acc') )
    indices.append( (dataset,'GM') )
    indices.append( (dataset,'FM') )
index = pd.MultiIndex.from_tuples(indices,names=['dataset','metric'])
X = np.vstack(colummns).T

df = pd.DataFrame(X,columns=name_columns,index=index)
print(df)

# %%
metric = 'Acc'
df_filter = (df
             .xs(metric,
                 level=1,
                 drop_level=False
                 )
            )
mean_cols = [col for col in df_filter.columns if 'mean' in col]
std_cols = [col for col in df_filter.columns if 'std' in col]



# df.to_excel('./SVM_krein.xlsx')
# %% Test Estadístico


di = {
    'glass1':'glass1',
    'ecoli-0_vs_1':'ecoli4',
    'wisconsin':'wisconsin',
    'pima':'pima',
    'glass0':'glass0',
    'yeast1':'yeast1',
    'haberman':'haberman',
    'vehicle2':'vehicle2',
    'vehicle3':'vehicle3',
    'glass-0-1-2-3_vs_4-5-6':'glass2',
    'vehicle0':'vehicle0',
    'ecoli1':'ecoli1',
    'new-thyroid1':'thyroid1',
    'new-thyroid2':'thyroid2',
    'ecoli2':'ecoli2',
    'yeast3':'yeast3',
    'ecoli3':'ecoli3',
    'yeast-2_vs_4':'yeast2'

}
#%%
# Nuevo diccionario con las correcciones
renombrar_datasets_corregido = {
    'glass1': 'glass1',
    'ecoli-0_vs_1': 'ecoli4',
    'wisconsin': 'wisconsin',
    'pima': 'pima',
    'glass0': 'glass0',
    'yeast1': 'yeast1',
    'haberman': 'haberman',
    'vehicle2': 'vehicle2',
    'vehicle3': 'vehicle3',
    'glass-0-1-2-3_vs_4-5-6': 'glass2',
    'vehicle0': 'vehicle0',
    'ecoli1': 'ecoli1',
    'new-thyroid1': 'thyroid1',
    'new-thyroid2': 'thyroid2',
    'ecoli2': 'ecoli2',
    'yeast3': 'yeast3',
    'ecoli3': 'ecoli3',
    'yeast-2_vs_4': 'yeast2'
}
df.reset_index(inplace=True)
#%%

metrics = df['metric'].unique()
dfs = {metric: df[df['metric'] == metric] for metric in metrics}

for metric, metric_df in dfs.items():
    # Cambiar los nombres de las bases de datos
    metric_df['dataset'] = metric_df['dataset'].map(di)
    # Ordenar según el orden de las keys en el diccionario
    metric_df['dataset'] = pd.Categorical(metric_df['dataset'], categories=di.values(), ordered=True)
    dfs[metric] = metric_df.sort_values('dataset')

plt.rcParams.update({'font.size': 15})

dfs_percent = {metric: metric_df.copy() for metric, metric_df in dfs.items()}
for metric_df in dfs_percent.values():
    mean_columns = [col for col in metric_df.columns if col.endswith('_mean')]
    for mean_col in mean_columns:
        metric_df[mean_col] *= 100  # Convertir a porcentaje

# Crear la figura con los ajustes solicitados
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8.5, 11))

# Trazar las gráficas de barras para cada métrica
for ax, (metric, metric_df) in zip(axes, dfs_percent.items()):
    # Extraer las columnas de interés
    mean_columns = [col for col in metric_df.columns if col.endswith('_mean')]
    std_columns = [col for col in metric_df.columns if col.endswith('_std')]

    # Número de algoritmos (columnas de media) y número de bases de datos
    n_algorithms = len(mean_columns)
    n_datasets = len(metric_df)

    # Posiciones de las barras para cada base de datos
    bar_positions = np.arange(n_datasets)

    # Ancho de cada barra
    bar_width = 0.9 / n_algorithms

    # Trazar las barras para cada algoritmo
    for i, (mean_col, std_col) in enumerate(zip(mean_columns, std_columns)):
        means = metric_df[mean_col].values
        stds = metric_df[std_col].values * 100  # Convertir desviaciones estándar a porcentaje
        method_name = mean_col.split('_mean')[0].replace('_TL1', '_{T_1}')
        ax.bar(bar_positions + i * bar_width, means, yerr=stds, width=bar_width, label=f'${method_name}$')

    # Configuraciones del gráfico
    ax.set_ylabel(f'${metric}$ (%)', fontsize=12, fontweight='bold')

    if ax == axes[-1]:  # Solo para la última subfigura
        ax.set_xticks(bar_positions + bar_width * (n_algorithms - 1) / 2 +0.3)
        ax.set_xticklabels(metric_df['dataset'], rotation=90, ha='right')
    else:
        ax.set_xticks([])

# Añadir una leyenda única en la parte inferior de la figura
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=n_algorithms, bbox_to_anchor=(0.5, -0.03))

# Ajustar las margenes y reducir los tamaños entre los subplots
plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1, hspace=0.1)

# Ajustar automáticamente los parámetros del subplot
plt.tight_layout()

# Mostrar la figura
plt.show()

fig.savefig("./imgs/All_performance_page_v2.pdf", bbox_inches='tight')

# %%
