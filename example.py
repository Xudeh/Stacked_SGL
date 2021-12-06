# -*- coding: utf-8 -*-
"""
Created on Sat May 29 10:22:53 2021

@author: Administrator
"""

from Stacked_sgl.s_sgl import Stacked_SGL

import numpy as np 
from numpy.random import shuffle

#%%
##example
def Calc_prob(y):
    p = np.exp(5*y) / (1 + np.exp(5*y))
    return p

def pred(predict):
    prediction = []
    for i in predict:
        if i>0.5:
            prediction.append(1)
        else:
            prediction.append(0) 
    return prediction

n_samples,n_features= 50,100  
m = 10 #group number 
l = 2 # generated group number 
np.random.seed(10) 
X = np.random.normal(size=[n_samples,n_features])
w1 = np.zeros(int(n_features/m))
for i in range(5):
    w1[i] = i+1
siga = np.random.randn(n_samples)#randn生成服从标准正态分布
y =  X[:,:10].dot(w1) + X[:,10:20].dot(w1) + 0.1*siga
w = np.hstack((np.hstack((w1,w1)),np.zeros(80)))
groups = np.arange(X.shape[1]) // 10  
p = Calc_prob(y)
clssify = pred(p)

idx_pos = []
idx_neg = []
for i in range(len(clssify)):
    if clssify[i] ==1:
        idx_pos.append(i)
    else:
        idx_neg.append(i)        
        
X1 = np.vstack((X[idx_pos,:],X[idx_neg,:]))  
"""original dataset [positive samples,negative samples]"""
np.random.seed(1) # shuffle the positive and negative samples respectively
shuffle(X1[:25])
shuffle(X1[25:])
Y = np.hstack((np.ones(25),np.zeros(25))) #generate labels of original dataset 

X_train = X1[5:45] #split training set
Y_train = Y[5:45] 
X_test = np.vstack((X1[:5,:],X1[45:,:])) # slpit independent test set
Y_test = np.hstack((Y[:5],Y[45:])) # label of independent test set 
x_pos = X_train[:20].copy()  # positive samples of training set 
x_neg = X_train[20:].copy() # negative samples of training set 

#%%

n = 5  # folds of cross validation
alpha_1 =[0.05,0.1] # aplha of base learners 
lambd_1 = np.arange(0.01,0.1,0.01).tolist() #regularization parameter of base learners
lambd_2 = np.arange(0.0005,0.1,0.0005).tolist() #regularization parameter of meter learner
S_S = Stacked_SGL(x_pos, x_neg,Y_train,X_test,Y_test,n,groups,
                  alpha = alpha_1,lambd = lambd_1,lambd2= lambd_2)
"""
x_neg: negative samples of training set
x_pos: postive samples of training set 
Y_train: labels of training set, labels of x_pos + labels of x_neg
X_test: independent test set
Y_test : labels od independent test set
n: folds of cross validation
alpha: corresponding aplha of each base learners
lambd: regularization parameter of base learners
lambd2: regularization parameter of meter learners
"""

C_M, A_L = S_S.Base() 
"""C_M: the training coefficient of the T base learners. 
   A_L: the optimal lambd and cross-validation sets of AUC
       and Acc for each alpha. for example [alpha, selected_lambd, auc, acc].
 """
Data, Data_test = S_S.Meta(A_L, C_M)
""" Data: P of the cross-validation set  probability matrix outputing 
           on the  base learner. n_train * T_base 
    Data_test: representing the independent test set probability matrix outputing 
              on the base learner. n_test * T_base
"""
y_pre, coef, Auc, Acc = S_S.Pre_MRL(Data, Data_test)  
""" y_pre : classification results of independent test set  
    coef: Estimation coefficient of meta learner.
    Auc: prediction AUC of independent test set
    Acc:prediction Acc of indenpendent test set
"""
lambd3 = 0.05
sgl_c = S_S.Stacked_hoc(C_M, A_L, coef, lambd3)
""" 
sgl_c:  Coefficient of post-hoc feature selection model
lambd3: Regularization parameters  of stacked SGL with post-hoc
       feature selection model,lambd3 can be adjusted to obtain different 
       degree of sparsity    
"""
idx = S_S.select_fea(sgl_c)         
""" idx: Index of the selected feature
"""