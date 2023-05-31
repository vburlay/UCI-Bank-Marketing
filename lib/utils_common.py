import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import  cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def probability(x_testcnn,y_test,model):
    probs = model.predict(x_testcnn)
    y_pred_df = pd.DataFrame(probs)
    y_pred_1 = y_pred_df.iloc[:, 1]
    y_test_df = pd.DataFrame(y_test)
# Put the index as ID column, remove index from both dataframes and combine them
    y_test_df["ID"] = y_test_df.index
    y_pred_1.reset_index(drop=True, inplace=True)
    y_test_df.reset_index(drop=True, inplace=True)
    y_pred_final = pd.concat([y_test_df, y_pred_1], axis=1)
    y_pred_final = y_pred_final.rename(columns={1: "YesProbabil", "y": "Yes/No"})
    y_pred_final = y_pred_final.reindex(["ID", "Yes/No", "YesProbabil"], axis=1)
    y_pred_final['Predicted'] = y_pred_final.YesProbabil.map(lambda x: 1 if x > 0.5 else 0)
        
    return y_pred_final
    
def log_reg_func():
    t = np.linspace(-10, 10, 100)
    sig = 1 / (1 + np.exp(-t))
    plt.figure(figsize=(9, 3))
    plt.plot([-10, 10], [0, 0], "k-")
    plt.plot([-10, 10], [0.5, 0.5], "k:")
    plt.plot([-10, 10], [1, 1], "k:")
    plt.plot([0, 0], [-1.1, 1.1], "k-")
    plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    plt.xlabel("t")
    plt.legend(loc="upper left", fontsize=20)
    plt.axis([-10, 10, -0.1, 1.1])
    plt.show()
def plot_roc_lr_xgb(fpr_xgb, tpr_xgb, fpr_lr, tpr_lr, label=None):
    fig,ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].plot(fpr_xgb, tpr_xgb, linewidth=2, label=label)
    ax[0].plot([0, 1], [0, 1], 'k--') # dashed diagonal
    ax[0].set_title('ROC Boosted trees - XGBoost',fontsize=18)
    ax[0].grid(True)
    ax[0].set_xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    ax[0].set_ylabel('True Positive Rate (Recall)', fontsize=16) # Not shown
    ax[1].plot(fpr_lr, tpr_lr, linewidth=2, label=label)
    ax[1].plot([0, 1], [0, 1], 'k--') # dashed diagonal
    ax[1].set_title('Logistic regression',fontsize=18)
    ax[1].grid(True)
    ax[1].set_xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    ax[1].set_ylabel('True Positive Rate (Recall)', fontsize=16) # Not shown
    plt.show()

def plot_acc_loss(cnnhistory, label=None):
    fig,ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].plot(cnnhistory['loss'], linewidth=2, label=label)
    ax[0].set_title('Model loss',fontsize=18)
    ax[0].grid(True)
    ax[0].set_xlabel('epoch', fontsize=16) # Not shown
    ax[0].set_ylabel('loss', fontsize=16) # Not shown
    ax[1].plot(cnnhistory['acc'], linewidth=2, label=label)
    ax[1].set_title('Model accurancy',fontsize=16)
    ax[1].grid(True)
    ax[1].set_xlabel('epoch', fontsize=16) # Not shown
    ax[1].set_ylabel('acc', fontsize=16) # Not shown
    plt.show()
    
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, model='clf', save=True):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, ax=ax, fmt="d", cmap=plt.cm.Oranges)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
        
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    b, t = plt.ylim() 
    b += 0.5 
    t -= 0.5 
    plt.ylim(b, t) 
    plt.show()

def ml_model(clf, X_train, X_test, y_train, y_test, save=False, print_stat=True, inc_train=False, cv=False):
   
     xgb_params = {'learning_rate': 0.01,
              'n_estimators': 100,
              'objective' : 'binary:logistic',
              'booster': 'gbtree',
              'scale_pos_weight':1}

     dt_params = {'criterion': 'entropy', 
             'max_depth': 35, 
             'min_samples_leaf': 4, 
             'min_samples_split': 23, 
             'max_leaf_nodes': 169}

     svc_params = {'kernel': 'poly',
              'degree': 3,
              'coef0' : 1,
              'C': 5 }

     knn_params = {'weights': 'distance', 
              'n_neighbors': 32}

     lr_params = {'multi_class':'ovr',
             'class_weight': None, 
             'solver': 'saga', 
             'max_iter':10000}


     model_abrv = {'dt':'Decision Tree Classifier', 
              'xgb':'Extreme Gradient Boosing',
              'svc':'Support Vector Machines',
              'kn':'K-Nearest Neighbors', 
              'lr':'Logistic Regression'}
     models =  {'dt':DecisionTreeClassifier(**dt_params),
               'xgb':XGBClassifier(**xgb_params),
               'svc':SVC(**svc_params), 
               'kn':KNeighborsClassifier(**knn_params),  
               'lr':LogisticRegression(**lr_params)}
     
     clf_model = models[clf]
     clf_model.fit(X_train, y_train)
     y_pred = clf_model.predict(X_test)
     if print_stat == True:
        clf_report = pd.DataFrame(classification_report(y_test,y_pred, output_dict=True)).T
        print(model_abrv[clf])
        print('\nTest Stats\n', classification_report(y_test,y_pred))
        print_confusion_matrix(confusion_matrix(y_test, y_pred), unique_labels(y_test, y_pred), model=clf)
        if inc_train == True:
            print(model_abrv[clf])
            print('\nTrain Stats\n', classification_report(y_train,clf_model.predict(X_train)))
            print_confusion_matrix(confusion_matrix(y_train, clf_model.predict(X_train)), unique_labels(y_test, y_pred), model=clf)
     if cv == True:
        print(model_abrv[clf] + ' CV Accuracy:',  
              np.mean(cross_val_score(clf_model, X_train, y_train, cv=5, scoring='accuracy')))
     if save == True:
        return clf_model
    
def ml_models_acc(X_train_reduced,X_test_reduced,y_train,y_test):
     xgb_params = {'learning_rate': 0.01,
              'n_estimators': 100,
              'objective' : 'binary:logistic',
              'booster': 'gbtree',
              'scale_pos_weight':1}

     dt_params = {'criterion': 'entropy', 
             'max_depth': 35, 
             'min_samples_leaf': 4, 
             'min_samples_split': 23, 
             'max_leaf_nodes': 169}

     svc_params = {'kernel': 'poly',
              'degree': 3,
              'coef0' : 1,
              'C': 5 }

     knn_params = {'weights': 'distance', 
              'n_neighbors': 32}

     lr_params = {'multi_class':'ovr',
             'class_weight': None, 
             'solver': 'saga', 
             'max_iter':10000}


     model_abrv = {'dt':'Decision Tree Classifier', 
              'xgb':'Extreme Gradient Boosing',
              'svc':'Support Vector Machines',
              'kn':'K-Nearest Neighbors', 
              'lr':'Logistic Regression'}
     models =  {'dt':DecisionTreeClassifier(**dt_params),
               'xgb':XGBClassifier(**xgb_params),
               'svc':SVC(**svc_params), 
               'kn':KNeighborsClassifier(**knn_params),  
               'lr':LogisticRegression(**lr_params)}
     for key in models.keys():
        ml_model(key, X_train_reduced,X_test_reduced,y_train,y_test,cv=True, print_stat=False)
        
def ml_models_eval(X_train_reduced,X_test_reduced,y_train,y_test):
     xgb_params = {'learning_rate': 0.01,
              'n_estimators': 100,
              'objective' : 'binary:logistic',
              'booster': 'gbtree',
              'scale_pos_weight':1}

     dt_params = {'criterion': 'entropy', 
             'max_depth': 35, 
             'min_samples_leaf': 4, 
             'min_samples_split': 23, 
             'max_leaf_nodes': 169}

     svc_params = {'kernel': 'poly',
              'degree': 3,
              'coef0' : 1,
              'C': 5 }

     knn_params = {'weights': 'distance', 
              'n_neighbors': 32}

     lr_params = {'multi_class':'ovr',
             'class_weight': None, 
             'solver': 'saga', 
             'max_iter':10000}


     model_abrv = {'dt':'Decision Tree Classifier', 
              'xgb':'Extreme Gradient Boosing',
              'svc':'Support Vector Machines',
              'kn':'K-Nearest Neighbors', 
              'lr':'Logistic Regression'}
    
     models =    {'dt':DecisionTreeClassifier(**dt_params),
              'xgb':XGBClassifier(**xgb_params),
               'svc':SVC(**svc_params), 
               'kn':KNeighborsClassifier(**knn_params),  
               'lr':LogisticRegression(**lr_params)}
     for key in models.keys():
         ml_model(key,X_train_reduced,X_test_reduced,y_train,y_test, inc_train=True)        
