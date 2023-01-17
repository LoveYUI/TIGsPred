from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix,classification_report
from imblearn.under_sampling import RandomUnderSampler
test_x = np.load("../test_x_new_reg.npy")
test_y = np.load("../test_y_new_reg.npy")
train_x = np.load("../train_x_new_reg.npy")
train_y = np.load("../train_y_new_reg.npy")
mid = np.quantile(train_y,0.75)
for i in range(len(train_y)):
    if train_y[i] > mid:
        train_y[i] = 1
    else:
        train_y[i] = 0
for i in range(len(test_y)):
    if test_y[i] > mid:
        test_y[i] = 1
    else:
        test_y[i] = 0
ros = RandomUnderSampler(random_state=0)
train_x, train_y = ros.fit_resample(train_x, train_y)
print(train_x.shape,train_y.shape)

model_svc = SVC(C=3,gamma=0.015625, kernel='rbf')
model_svc.fit(train_x, train_y)
pp = model_svc.predict(test_x)
print(classification_report(test_y,model_svc.predict(test_x)))

mid = np.quantile(train_y,0.5)
for i in range(len(train_y)):
    if train_y[i] > mid:
        train_y[i] = 1
    else:
        train_y[i] = 0
for i in range(len(test_y)):
    if test_y[i] > mid:
        test_y[i] = 1
    else:
        test_y[i] = 0

#svr
model_svc = SVC(C=3, gamma=0.0078125, kernel='rbf')
model_svc.fit(train_x, train_y)
print(classification_report(test_y,model_svc.predict(test_x)))