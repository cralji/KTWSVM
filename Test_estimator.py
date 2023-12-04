#%%
from TWSVM_Krein.estimators import TWSVM_krein
from TWSVM_Krein.kernels import tanh_kernel
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
#%%
path_dataset = 'data\haberman.dat'

data = pd.read_table(path_dataset,delimiter=',',header=None).to_numpy()
# Data
X,t = data[:,0:-1].astype(np.float32),data[:,-1].astype(str)
X = StandardScaler().fit_transform(X)
labels = np.unique(t)

nC = labels.size
n_per_class = [sum(t==labels[i]) for i in range(nC)]
# who_is_minoritary = np.argmin(n_per_class)
who_is_majority = np.argmax(n_per_class)
y = np.ones_like(t)
y[t==labels[who_is_majority]] = -1
y = y.astype(np.int8)
# %%
kernel = tanh_kernel(gamma = 1e-1)
clf = TWSVM_krein(kernel = kernel)
clf.fit(X,y)
# %%
t_est = clf.predict(X)
print(classification_report(y,t_est))