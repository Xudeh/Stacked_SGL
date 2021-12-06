# Stacked-SGL
* Stacked SGL uses a machine learning method by combining sparse group lasso and stacking for cancer classification. Users can run program with specified cancer gene expression data. 
## Installation
* It is required to install the following dependencies in order to be able to run the files: For the Stacked SGL method
```
python >= 3.5
numpy = 1.18.4
pandas = 0.23.4
joblib = 1.0.0
scipy >= 1.4.1
scikit-learn >= 0.23.2
```
```
git clone https://github.com/huanheaha/Stacked-SGL.git
python setup.py install
```
# Running the Code
* Download all the data and code in Stacked-SGL-master to the local address.
* Open Python editor， then dictory to stacked-SGL-master folder which contains example.py. 
* Make sure the data and code are in one folder, or enter the exact data address when you run the code.
## **An example of applying the Stacked SGL is provided in example.py.** <br>
Running the example in example.py. Specific example including parameter description and parameter selection is provided in this file.
```
S_S = Stacked_SGL(x_pos, x_neg,Y_train,X_test,Y_test,n,groups,
                  alpha = alpha_1,lambd = lambd_1,lambd2= lambd_2)
```
Prediction results will show in the terminal.
# Announcements
* The input data set should be standardized.
* The csv files of LIHC.csv, THCA.csv and Lung.csv represent the mRNA gene expression data of three cancer genes trained in the model. The LIHC-groups.csv, THCA-groups.csv and Lung-groups.csv include gene pathway information of pathway name and number of genes in the pathway.
* The optimal parameters are trained in LIHC_best.model, THCA_best.model and Lung_best.model.You can run predict.py to obtain the prediction results of real data.
