import numpy as np
from utils import no_quotation,train_and_test
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.svm import SVR,SVC

train_x, train_y = np.load("train_x_new.npy"),np.load("train_y_new.npy")
test_x, test_y = np.load("test_x_new.npy"),np.load("test_y_new.npy")

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_y.reshape([train_y.shape[0],1]))
train_y = scaler.transform(train_y.reshape([train_y.shape[0],1])).reshape([train_y.shape[0]])
test_y = scaler.transform(test_y.reshape([test_y.shape[0],1])).reshape([test_y.shape[0]])

print(train_x.shape,train_y.shape)
svm_model = SVR()
svm_model.fit(train_x,train_y)
# svm_model.fit(np.log(train_x),np.log(train_y))
print(svm_model.score(test_x,test_y))