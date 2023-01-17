from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import numpy as np
import joblib

train_x = np.load("train_x.npy")
train_y = np.load("train_y.npy")
test_x = np.load("test_x.npy")
test_y = np.load("test_y.npy")
print(train_x.shape,train_y.shape)
#svr-linear
model_svr = SVR(kernel='linear')
model_svr.fit(train_x, train_y)
print(model_svr.score(test_x,test_y))

#svr
model_svr = SVR(C=1.5, gamma=0.022, kernel='rbf')
model_svr.fit(train_x, train_y)
print(model_svr.score(test_x,test_y))
joblib.dump(model_svr,"model_svr.save")
exit()
#rf
model_rf = RandomForestRegressor(n_estimators=400,min_samples_leaf=4,max_features=250)
model_rf.fit(train_x, train_y)
print(model_rf.score(test_x,test_y))

#xgboost
model_xgb = XGBRegressor(n_estimators=150,min_child_weight=32,max_depth=7)
model_xgb.fit(train_x,train_y)
print(model_xgb.score(test_x,test_y))


#decision tree
model_dt = DecisionTreeRegressor(min_samples_leaf=4)
model_dt.fit(train_x,train_y)
print(model_dt.score(test_x,test_y))

#logistic reg
# model_log = LogisticRegression()
# model_log.fit(train_x,train_y)
# print(model_log.score(test_x,test_y))

#linear reg
model_lin = LinearRegression()
model_lin.fit(train_x,train_y)
print(model_lin.score(test_x,test_y))

#Ridge
model_rig = Ridge()
model_rig.fit(train_x,train_y)
print(model_rig.score(test_x,test_y))

#Lasso
model_las = Lasso()
model_las.fit(train_x,train_y)
print(model_las.score(test_x,test_y))

#KNN
model_knn = KNeighborsRegressor(n_neighbors=20)
model_knn.fit(train_x,train_y)
print(model_knn.score(test_x,test_y))