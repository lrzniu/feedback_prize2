import numpy as np
from sklearn import metrics
import torch.nn.functional as F
import sklearn
t_true = np.random.randint(0, 2, size=1000)

# 模型1
pred1 = t_true + np.random.randn(1000)
pred1 = np.clip(pred1, 0, 1)

# 模型2
pred2 = t_true + np.random.randn(1000) - 0.2
pred2 = np.clip(pred2, 0, 1)

# 模型3
pred3 = t_true + np.random.randn(1000) - 0.1
pred3 = np.clip(pred3, 0, 1)
pred1=pred1**2
def aucfun(act, pred):
    fpr, tpr, thresholds = metrics.roc_curve(act, pred, pos_label=1)
    return metrics.auc(fpr, tpr)




# print(aucfun(t_true, pred1))
# print(aucfun(t_true, pred2))
# print(aucfun(t_true, pred3))
from scipy.stats import rankdata

# array([1., 3., 2.])
print(rankdata([1,2.5,2,4]))

p = np.random.rand(2,2)

p=p/p.sum(axis=1,keepdims=1)
print(p)
x=0.8
a=2
print(x)
print(x**a/(x**a+(1-x)**a))