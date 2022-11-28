#%% Libraries
from sklearn.metrics.pairwise import sigmoid_kernel
from scipy.spatial.distance import cdist
from numpy import maximum
#%%
class tanh_kernel:
    def __init__(self,
                gamma=None,
                coef0=1):
        self.gamma = gamma
        self.coef0 = coef0
    
    def __call__(self,X,Y):
        K = sigmoid_kernel(X,Y,gamma = self.gamma,coef0 = self.coef0)
        return K
#%% TL1 kernel https://doi.org/10.1016/j.acha.2016.09.001
class TL1:
    def __init__(self,p=1):
        self.p = p
    def __call__(self, X,Y):
        D = cdist(X,Y,'minkowski',p=1)
        K = maximum(1-D,0)
        return K