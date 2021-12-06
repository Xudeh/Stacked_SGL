# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:20:42 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 21 17:08:33 2021

@author: Administrator
"""

from sklearn import linear_model     
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectFromModel
import numpy as np 
from sklearn.metrics import roc_auc_score
from sklearn import metrics
# from sgl_logistic import sparse_group_lasso
# from sgl_logistic import calc_prob
import math
import numpy as np
from scipy import linalg
from sklearn.metrics import roc_auc_score
from numpy import mean
import joblib

# 真实数据 THCA
import pandas as pd
import numpy as np
from sklearn import preprocessing
from numpy.random import shuffle
# 真实数据 THCA
def read_Lung():    
    data = pd.read_csv("Lung.csv",sep = ",",index_col=0) 
    gene_name = data.index.tolist()
    data_scaler = preprocessing.scale(np.array(data).T) #标准化后的数据
    y = np.hstack((np.ones(501),np.zeros(513))) 
    np.random.seed(1) #1
    np.random.shuffle(data_scaler[:501])  #打乱正样本
    np.random.shuffle(data_scaler[501:])

    X_Train = data_scaler[150:850,:]
    Y_train = y[150:850]
    X_test = np.vstack((data_scaler[:150,:],data_scaler[850:,:])) #独立测试集
    Y_test = np.hstack((y[:150],y[850:]))
    x_pos = X_Train[:350].copy()  #打乱正负样本，正样本30，负样本30
    np.random.seed(1) #1
    shuffle(x_pos)
    x_neg = X_Train[350:].copy()
    np.random.seed(2) #2
    np.random.shuffle(x_neg)
    
    set_size = pd.read_csv("Lung-groups.csv",sep = ",") 
    s = set_size['size_hoc'].tolist()
    groups = []
    for i in range(len(s)):
        a = [i] * int(s[i])
        groups += a
    return x_pos, x_neg, X_test, Y_train,Y_test, groups,gene_name



x_pos,x_neg, X_test,Y_train,Y_test, groups,gene_name = read_Lung()
n = 5     #表示5折
m = len(x_pos)//n #每折的正负样本数,12
alpha_values = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,
                0.7,0.75,0.8,0.85,0.9,0.95]
lambda_values = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,
                 0.01,0.01,0.01,0.01,0.01]
a_l = []
for i in range(len(alpha_values)):
    a_l.append([alpha_values[i],lambda_values[i]])
    
lambd2 = 0.00034
    
MAX_ITER = 1000

def soft_threshold(a, b):
    # vectorized version
    return np.sign(a) * np.fmax(np.abs(a) - b, 0)

def calc_prob( X, betas):
    power = X.dot(betas)
    p = np.exp(power) / (1 + np.exp(power))
    return p

def pred(predict):
    prediction = []
    for i in predict:
        if i>0.5:
            prediction.append(1)
        else:
            prediction.append(0) 
    return prediction

def sparse_group_lasso(X, y, alpha, rho, groups, max_iter= MAX_ITER, rtol=1e-6,
                verbose=False):
    """
    Linear least-squares with l2/l1 + l1 regularization solver.

    Solves problem of the form:

    (1 / (2 n_samples)) * ||Xb - y||^2_2 +
        [ (alpha * (1 - rho) * sum(sqrt(#j) * ||b_j||_2) + alpha * rho ||b_j||_1) ]

    where b_j is the coefficients of b in the
    j-th group. Also known as the `sparse group lasso`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Design Matrix.

    y : array of shape (n_samples,)
    rho: rho = 0,group lasso.
        rho = 1,lasso
    alpha : float or array
        Amount of penalization to use.

    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.

    rtol : float
        Relative tolerance. ensures ||(x - x_) / x_|| < rtol,
        where x_ is the approximate solution and x is the
        true solution. TODO duality gap

    Returns
    -------
    x : array
        vector of coefficients

    References
    ----------
    "A sparse-group lasso", Noah Simon et al.
    """
    # .. local variables ..
    X, y, groups, alpha = map(np.asanyarray, (X, y, groups, alpha))
    if groups.shape[0] != X.shape[1]:
        raise ValueError('Groups should be of shape %s got %s instead' % ((X.shape[1],), groups.shape))
    w_new = np.zeros(X.shape[1], dtype=X.dtype)
    n_samples = X.shape[0]
    alpha = alpha * n_samples

    # .. use integer indices for groups ..
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    Xy = np.dot(X.T, y)
    K = np.dot(X.T, X)
    step_size = 1. / (linalg.norm(X, 2) ** 2)
    _K = [K[group][:, group] for group in group_labels]

    for n_iter in range(max_iter):
        w_old = w_new.copy()
        perm = np.random.permutation(len(group_labels))
        X_residual = Xy - np.dot(K, w_new) # could be updated, but kernprof says it's peanuts Xy-T_T*X*w
        for i in perm:
            group = group_labels[i]
            #import ipdb; ipdb.set_trace()
            p_j = math.sqrt(group.size)
            Kgg = _K[i]
            X_r_k = X_residual[group] + np.dot(Kgg, w_new[group]) #X*r(-k)
            s = soft_threshold(X_r_k, alpha * rho)
            # .. step 2 ..
            if np.linalg.norm(s) <= (1 - rho) * alpha * p_j:
                w_new[group] = 0.
            else:
                # .. step 3 ..
                for _ in range(2 * group.size): # just a heuristic
#                    grad_l =  - (X_r_k - np.dot(Kgg, w_new[group]))
                    Pro  = calc_prob( X, w_new)
                    grad_l = - np.dot(X[:,group].T ,(y - Pro))
                    tmp = soft_threshold(w_new[group] - step_size * grad_l, step_size * rho * alpha)
                    tmp *= max(1 - step_size * p_j * (1 - rho) * alpha / np.linalg.norm(tmp), 0)
                    delta = linalg.norm(tmp - w_new[group])
                    w_new[group] = tmp
                    if delta < 1e-3:
                        break

                assert np.isfinite(w_new[group]).all()

        norm_w_new = max(np.linalg.norm(w_new), 1e-10)
        if np.linalg.norm(w_new - w_old) / norm_w_new < rtol:
            #import ipdb; ipdb.set_trace()
            break
    return w_new

def meta_Pred(y_p):  #元学习器预测结果进行分类
    y_pred = []
    for j in range(len(y_p)):
        if y_p[j][0] > y_p[j][1]:
            y_pred.append(1)
        else :
            y_pred.append(0)
    return y_pred



def calc_lambd(x_pos,x_neg,Y_train,m,n,a_l,groups):
    alpha_lambd = []
    Coef_modle = []
    for i in a_l:
        alpha = i[0]
        lambd = i[1]
        perfor_auc = [] ; 
        coef_val = [];
        # for lambd in lambda_values:    
        ACC = []; AUC = [];
        coef = []; 
        for k in range(n):  #10折,每折总24个样本，正负12个
            x_val = np.vstack((x_pos[k * m:(k+1)*m],x_neg[k*m:(k+1)*m]))
            x_pos_train = np.vstack((x_pos[:k*m], x_pos[(k+1)*m:]))
            x_neg_train = np.vstack((x_neg[:k*m], x_neg[(k+1)*m:]))
            x_train = np.vstack((x_pos_train,x_neg_train)) #训练集，用来训练基学习器,48
            y_val = Y_train[(n-1)*m: (n+1)*m]  #验证集，用于验证基学习器并生成元学习器的数据
            y_train = Y_train[m: (2*n-1)*m] 
                
            coefs_train = sparse_group_lasso(x_train,y_train, lambd,
                                                alpha, groups, max_iter= 2000,verbose=True)#基学习器
            pred_val =  calc_prob(x_val, coefs_train) #计算测试集概率
            predict_val = pred(pred_val) #计算分类为0或1
                
            acc_score = metrics.accuracy_score(y_val, predict_val) #acc
            auc_score = roc_auc_score(y_val, pred_val)
            coef.append(coefs_train) #每个lambda对应的五折训练结果
            ACC.append(acc_score)
            AUC.append(auc_score)
          
        coef_val = coef_val + coef  #保留所有lambda训练结                   
        perfor_auc.append([mean(AUC),mean(ACC),lambd]) #每次五折验证集auc的平均值
        Coef_modle.append(coef_val)  #选出平均acc最高的lambda对应的结果
        alpha_lambd.append([alpha, perfor_auc[0][2], perfor_auc[0][0],perfor_auc[0][1]]) 
    return Coef_modle,alpha_lambd

class Stacked_SGL():
    def __init__(self, x_pos, x_neg, Y_train,m,n, A_L,C_M, groups, gene_name,lambd2):
        self.x_pos= x_pos
        self.x_neg = x_neg
        # self.X_test = X_test 
        self.Y_train = Y_train
        # self.Y_test = Y_test
        self.A_L = A_L
        self.C_M = C_M
        self.m = m
        self.n = n
        self.groups = groups
        self.gene_name = gene_name
        self.lambd2 = lambd2
        
    # def calc_lambd(self):
    # # def calc_lambd(x_pos,x_neg,Y_train,m,n,a_l,groups):
    #     alpha_lambd = []
    #     Coef_modle = []
    #     for i in self.a_l:
    #         alpha = i[0]
    #         lambd = i[1]
    #     # for alpha in alpha_values:  #不同的alpha,mix参数
    #         perfor_auc = [] ; 
    #         coef_val = [];
    #         # for lambd in lambda_values:    
    #         ACC = []; AUC = [];
    #         coef = []; 
    #             # perfor_auc = []; coef_val = [];FEA_val = []
    #         for k in range(n):  #10折,每折总24个样本，正负12个
    #             x_val = np.vstack((self.x_pos[k * m:(k+1)*m],self.x_neg[k*m:(k+1)*m]))
    #             x_pos_train = np.vstack((self.x_pos[:k*m], self.x_pos[(k+1)*m:]))
    #             x_neg_train = np.vstack((self.x_neg[:k*m], self.x_neg[(k+1)*m:]))
    #             x_train = np.vstack((x_pos_train,x_neg_train)) #训练集，用来训练基学习器,48
    #             y_val = self.Y_train[(self.n-1)*self.m : (self.n + 1)*self.m]  #验证集，用于验证基学习器并生成元学习器的数据
    #             y_train = self.Y_train[self.m: (2*self.n -1) * self.m] 
                    
    #             coefs_train = sparse_group_lasso(x_train,y_train, lambd,
    #                                 alpha, self.groups, max_iter= 2000,verbose=True)#基学习器
    #             pred_val =  calc_prob(x_val, coefs_train) #计算测试集概率
    #             predict_val = pred(pred_val) #计算分类为0或1
                    
    #             acc_score = metrics.accuracy_score(y_val, predict_val) #acc
    #             auc_score = roc_auc_score(y_val, pred_val)
    #             coef.append(coefs_train) #每个lambda对应的五折训练结果
    #             ACC.append(acc_score)
    #             AUC.append(auc_score)
              
    #         coef_val = coef_val + coef  #保留所有lambda训练结                   
    #         perfor_auc.append([mean(AUC),mean(ACC),lambd]) #每次五折验证集auc的平均值
    #         Coef_modle.append(coef_val)  #选出平均acc最高的lambda对应的结果
    #         alpha_lambd.append([alpha, perfor_auc[0][2], perfor_auc[0][0],perfor_auc[0][1]]) 
    #     return Coef_modle,alpha_lambd
        
    def calc_MRL(self, X_test, Y_test):
    # def calc_meta(alpha_values,Coef_modle,X_test,Y_test,x_pos,x_neg,Y_train,n,m):
        meta_test = []
        meta_data = []
        test_AUC_ACC = []
        # C_M, A_L = self.calc_lambd()
        for i in range(len(self.A_L)):
            select_coef = self.C_M[i]
            meta_train_data = []  
            meta_test_data = []
            for k in range(self.n): #五折交叉验证选定最优参数后，再训练训练集，得到基学习器参数
                x_val = np.vstack((self.x_pos[k *self.m:(k+1)* self.m],self.x_neg[k*self.m:(k+1)*self.m]))   
                y_val = self.Y_train[(self.n-1)*self.m: (self.n+1)*self.m]
                
                p_train = calc_prob(x_val, select_coef[k]) #输出验证集预测值
                p_test = calc_prob(X_test, select_coef[k]) #输出独立测试集预测值   
                meta_train_data = meta_train_data + p_train.tolist()
                meta_test_data.append(p_test)
            
            meta_data.append(meta_train_data)  #得到所有基学习器的验证集结果，即元学习器的训练集
            test_data = np.mean(np.array(meta_test_data), axis= 0) #独立测试集取五次结果平均值
            meta_test.append(test_data)       #得到所有基学习器的独立测试集结果
            test_acc = metrics.accuracy_score(Y_test, pred(test_data))
            test_auc = roc_auc_score(Y_test, test_data)
            test_AUC_ACC.append([test_auc,test_acc])
        
        data2_test = np.array(meta_test).T  #独立测试
        data2 = np.array(meta_data).T   #元学习器训练集，得到的数据是6正6负+6正6负拼接
        # return data2, data2_test, test_AUC_ACC
    # def calc_MRL(self):       
    # def calc_MRL(lambd2, data2,data2_test,m,n,Y_test):
        y1 = np.tile(np.hstack((np.ones(self.m),np.zeros(self.m))), self.n)     #生成元学习器数据label
        y2 = np.tile(np.hstack((np.zeros(self.m),np.ones(self.m))), self.n)           
        y_meta = np.vstack((y1,y2)).T 
        Data2 = np.hstack((data2, y_meta)) 
        
        clf1 = linear_model.MultiTaskLasso(alpha = lambd2,max_iter= 50000).fit(Data2[:,:-2],Data2[:,-2:]) 
        y_test = clf1.predict(data2_test)
        coef = np.vstack((clf1.intercept_, clf1.coef_.T))
        # y_pre = meta_Pred(y_test)
        return y_test, coef 
        
    def Pred(self,X_test,Y_test):
        y_test, c = self.calc_MRL(X_test,Y_test)
        y_pre = meta_Pred(y_test)
        return y_pre 
        
    def scores(self,X_test,Y_test):
        y_test, c = self.calc_MRL(X_test, Y_test)
        Acc = metrics.accuracy_score(Y_test, meta_Pred(y_test))
        Auc = roc_auc_score(Y_test,y_test[:,0])
        # coef2 = np.vstack((clf1.intercept_, clf1.coef_.T))
        return Auc, Acc 
    
    # def select(self):
    def Stacked_hoc(self, X_test,Y_test):
        y_test, coef2 = self.calc_MRL(X_test,Y_test)
    # def post_hoc(Coef_model,x_pos,x_neg,coef2): #测试集训练后再预测的结果
        X_hoc = []
        for i in range(len(self.C_M)):      #基模型的个数
            selection_coef = self.C_M[i]
            Train = []
            for j in range(self.n):
                p_T = calc_prob(np.vstack((self.x_pos, self.x_neg)),selection_coef[j])
                Train.append(p_T)
            T_data = np.mean(np.array(Train), axis= 0) #独立测试集取五次结果平均值
            X_hoc.append(T_data)   
        X_hoc = np.array(X_hoc).T
        y_hoc = np.dot(X_hoc, coef2[1:,0]) + coef2[0,0]
        
        sgl_coefs = sparse_group_lasso(np.vstack((self.x_pos,self.x_neg)), pred(y_hoc),0.01,
                                         0.8, self.groups, max_iter= 2000,verbose=True) #lambda越大越稀疏
        # predict = pred(calc_prob(X_test, sgl_coefs))#计算独立测试集            
        # sgl_acc = metrics.accuracy_score(Y_test, predict) #
        # sgl_auc = roc_auc_score(Y_test, calc_prob(X_test, sgl_coefs))
        return sgl_coefs     
    
    def select_fea(self,sgl_coef):
        idx = []
        for i in range(len(sgl_coef)):
            if (sgl_coef[i] != 0):
                idx.append([i]) 
        return idx
    
    def Extract(self, num,X_test,Y_test):
        sgl_coefs = self.Stacked_hoc(X_test,Y_test)
        coef = np.maximum(sgl_coefs,-sgl_coefs)
        coef = list(coef)
        c_sorted = sorted(enumerate(coef), key=lambda x:x[1])  
        c_list = []
        for i in range(len(c_sorted)):
            x = list(c_sorted[i])
            c_list.append(x)
        s_g = c_list[-num:]
        s = [i[0] for i in s_g]
        gene_select = []
        for i in s:
            gene_select.append(self.gene_name[i])  #lasso选出基因名
        return s, gene_select


