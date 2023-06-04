# **Bank Marketing** 

![image](https://github.com/vburlay/uci-bank-marketing/raw/master/images/dataset-cover.png)
> A study of the customer database for the purpose of finding and analyzing the potential customers.
> Live demo [_here_](https://vburlay-uci-bank-marketing-streamlit-demo-sbf8b9.streamlit.app/).

## Table of Contents
* [Genelal Info](#general-nformation)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)


## General Information
> In this project, a major goal was to create a current model. This current model should help to give the answers to two research questions. On the one hand, the model should identify the "potential customers", on the other hand, it should find a group of similar customers.
 > Data set: "Caravan Insurance Challenge" comes from Kaggle [_here_](https://archive.ics.uci.edu/ml/datasets/bank+marketing).


## Technologies Used
- Python - version 3.8.0
- Jupyter notebook - version 1.0


## Features
- Machine Learning (Logistic regression, XGBoos, Decision Tree, Random Forest, SVC, KNN, PCA, Clustering)
- Deep Learning (CNN)

## Screenshots
* **ROC (Logistic regression + XGBoost )** 
![image1](https://github.com/vburlay/uci-bank-marketing/raw/master/images/ROC.PNG) 

| Architecture    | Accuracy of Training data | Accuracy of Test data |
|-----------|:-------------------------:|----------------------:|
|Decision Tree Classifier  |           0,88            |                  0,94 |
|Extreme Gradient Boosing  |           0,89            |                  0,94 |
|Support Vector Machines  |           0,89            |                  0,89 |
|K-Nearest Neighbors  |           0,88            |                   1.0 |
|Logistic Regression  |           0,89            |                  0,89 |
|Convolutional neural network (CNN) |           0,90            |                  0,90 |


* **CNN (Architecture)**

![image2](https://github.com/vburlay/uci-bank-marketing/raw/master/images/cnn.PNG) 

* **CNN (Evaluation)**
![image3](https://github.com/vburlay/uci-bank-marketing/raw/master/images/loss_acc.PNG) 

### Predicted
![image4](https://github.com/vburlay/uci-bank-marketing/raw/master/images/pred.PNG)
### Origin
![image5](https://github.com/vburlay/uci-bank-marketing/raw/master/images/yes_no.PNG)

## Setup
You can install the package as follows:
```r
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.colors as mc
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
plt.style.use('./deeplearning.mplstyle')
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import  cross_val_predict
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
import keras
from lib.utils_common import *
from lib.cnn_keras import build_model, k_fold
from joblib import dump, load
from pathlib import Path  
#CNN
from keras.models import Sequential
import keras
from keras.layers import Dense
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,SpatialDropout1D
from keras.models import Model
```


## Usage
The result 0.94025 - 94 % is good but with preprocessing by clustering the accuracy can be improved. Clustering (K-Means) can be an efficient approach for dimensionality reduction but for this a pipeline has to be created that divides the training data into clusters 34 and replaces the data by their distances to this cluster 34 to apply a logistic regression model afterwards:
```r
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters = d)),
    ("log_reg", LogisticRegression(multi_class = 'ovr',
             class_weight = None, 
             solver= 'saga', 
             max_iter = 10000)),
])
```


## Project Status
Project is: _complete_ 


## Room for Improvement

- The data implemented in the analysis has a relatively small volume. This should be improved by the new measurements of the characteristics.
- It is also conceivable that the further number of new customer groups will be included in the analysis. In this way, the new characteristics of customers can make the results more meaningful.



## Contact
Created by [Vladimir Burlay](wladimir.burlay@gmail.com) - feel free to contact me!



