# This is a sample Python script.
import os
from joblib import  load
from lib.utils_common import probability
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import keras
# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def dl(path_dir, X_data, y_data):
    # load Date
    model_cnn = keras.models.load_model(path_dir)
    df_cnn_res = probability(X_data, y_data, model_cnn)
    return df_cnn_res
def lr(path_lr,X_reduced, y_data):
    # load Date
    log_reg = load(path_lr)
    y_lr_pred = probability(X_reduced, y_data, log_reg)
    return y_lr_pred
def xgb_boost(path_xgb,X_reduced, y_data):
    # load Date
    xgb_clf = load(path_xgb)
    y_xgb_pred = probability(X_reduced, y_data, xgb_clf )
    return y_xgb_pred


def data_ml_prep():
    data = pd.read_csv("data/bank-additional-full.csv", sep=";", keep_default_na=False)
    y_data = data['y']
    X_data = data.drop(['y'], axis=1)
    X_data = pd.get_dummies(X_data)  # Encoding
    mm_scale = preprocessing.MinMaxScaler()  # Scaling
    X_data[X_data.columns] = mm_scale.fit_transform(X_data[X_data.columns])
    def y_val(x):
        if x == 'no':
            return 0
        elif x == 'yes':
            return 1
    y_data = y_data.apply(y_val)
    #PCA
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(X_data)

    return X_reduced,X_data, y_data



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path_lr = os.getcwd() + '\models\log_reg.joblib'
    path_xgb = os.getcwd() + '\models\log_xgb.joblib'
    path_dir = os.getcwd() + '\models\cnn_model.h5'

    X_reduced, X_data, y_data = data_ml_prep()
#    y_lr_pred = lr(path_lr,X_reduced, y_data)
#    y_xgb_pred = xgb_boost(path_xgb,X_reduced, y_data)
    y_cnn_pred = dl(path_dir,X_data, y_data)
    y_cnn_pred


