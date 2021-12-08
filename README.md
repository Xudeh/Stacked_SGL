# Stacked-SGL
* Stacked SGL uses a machine learning method by combining sparse group lasso and stacking for cancer classification. Users can run program with specified cancer gene expression data. 
## Installation
* It is required to install the following dependencies in order to be able to run the package: For the Stacked SGL method of the package named Stacked_sgl
```
python >= 3.5
numpy = 1.18.4
pandas = 0.23.4
joblib = 1.0.0
scipy >= 1.4.1
scikit-learn >= 0.23.2
```
You can also clone the repository and do a manual install.
```
git clone https://github.com/huanheaha/Stacked-SGL.git
python setup.py install
```
# Running the Code
* Acquire all the data and code in Stacked-SGL to the local address.
* Open Python editor,then dictory to stacked-SGL folder which contains example.py. 
* Make sure the data and code are in one folder,or enter the exact data address when you run the code.
## **An example of applying the Stacked SGL is provided in example.py.** <br>
Running the example in example.py. Specific example including parameter description and parameter selection is provided in this file.
```
from Stacked_sgl.s_sgl import Stacked_SGL
S_S = Stacked_SGL(x_pos, x_neg,Y_train,X_test,Y_test,n,groups,
                  alpha = alpha_1,lambd = lambd_1,lambd2= lambd_2)
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
"""      
```
Prediction results will show in the terminal.
```
print(y_pre)
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
```
In addition,some optional parameters are also available:

*```alpha```: mixing parameter (default [0.05,0.1])
*lambd: regularization parameter of base learners(default {0.01*n|n=1,...,100})
*lambd2: regularization parameter of meter learners(default {0.0005*n|n=1,...,200}
*lambd3: regularization parameters  of stacked SGL with post-hoc
       feature selection model
```
The parameters can be adjusted according to user requirements.
# Announcements
* The input data set should be standardized.
* The csv files of LIHC.csv, THCA.csv and Lung.csv represent the mRNA gene expression data of three cancer genes trained in the model. The LIHC-groups.csv, THCA-groups.csv and Lung-groups.csv include gene pathway information of pathway name and number of genes in the pathway.
* The optimal parameters are trained in LIHC_best.model, THCA_best.model and Lung_best.model.You can run predict.py to obtain the prediction results of real data.
