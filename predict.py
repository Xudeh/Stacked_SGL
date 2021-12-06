# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:58:55 2021

@author: Administrator
"""

#THCA dataset
from THCA import read_THCA
from THCA import Stacked_SGL
import joblib
x_pos,x_neg, X_test,Y_train,Y_test, groups,gene_name = read_THCA()
"""
x_pos: 40 turmor samples of THCA train set.
x_neg：40 normal samples of THCA train set.
X_test: 48 independent test samples.
Y_train: label of 80 train samples. 
Y_test: label of 48 independent test samples.
groups: 188 gene pathways of THCA gene-expression data.
gene_name: 3073 gene names of THCA gene-expression data. 
"""
model1 = joblib.load(filename="THCA_best.model")
y_pre = model1.Pred(X_test, Y_test)
"""
y_pre: classification results of independent test set of THCA data.
"""
idx , select_gene = model1.Extract(100,X_test,Y_test)
"""
idx: indexs of selected gene.
select_gene: genes name of selected genes.
"""

#%%
#LIHC dataset 
from LIHC import read_LIHC
from LIHC import Stacked_SGL
import joblib
model2 = joblib.load(filename="LIHC_best.model")
x_pos,x_neg, X_test,Y_train,Y_test, groups,gene_name = read_LIHC()
"""
x_pos: 30 turmor samples of LIHC train set.
x_neg：30 normal samples of LIHC train set.
X_test: 40 independent test samples.
Y_train: label of 60 train samples. 
Y_test: label of 40 independent test samples.
groups: 192 gene pathways of LIHC gene-expression data.
gene_name: 5183 gene names of LIHC gene-expression data. 
"""
y_pre = model2.Pred(X_test, Y_test)
"""
y_pre: classification results of independent test set of LIHC data.
"""

#%%
#Lung cancer dataset
from Lung import read_Lung
from Lung import Stacked_SGL
model3 = joblib.load(filename="Lung_best.model")
x_pos,x_neg, X_test,Y_train,Y_test, groups,gene_name = read_Lung()
"""
x_pos: 350 LUSC samples of Lung train set.
x_neg：350 LUAD samples of Lung train set.
X_test: 314 independent test samples.
Y_train: label of 700 train samples. 
Y_test: label of 314 independent test samples.
groups: 230 gene pathways of Lung gene-expression data.
gene_name: 4874 gene names of LIHC gene-expression data. 
"""
y_pre = model3.Pred(X_test, Y_test)
"""
y_pre: classification results of independent test set of Lung data.
"""








