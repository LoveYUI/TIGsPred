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
from scipy.stats import spearmanr,pearsonr,wilcoxon,mannwhitneyu
import joblib

test_x = np.load("../test_x_new_reg.npy")
test_y = np.load("../test_y_new_reg.npy")
train_x = np.load("../train_x_new_reg.npy")
train_y = np.load("../train_y_new_reg.npy")

name = ['lin','rbf','rf','xgboost','dt','lin-reg','Ridge','lasso','knn']
test_result = []

#svr-linear
model_svr = SVR(kernel='linear')
model_svr.fit(train_x, train_y)
pred_y = model_svr.predict(test_x)
test_result.append([model_svr.score(test_x,test_y),spearmanr(pred_y,test_y),pearsonr(pred_y,test_y)])

#svr
model_svr = SVR(C=3, gamma=0.03125, kernel='rbf')
model_svr.fit(train_x, train_y)
pred_y = model_svr.predict(test_x)
test_result.append([model_svr.score(test_x,test_y),spearmanr(pred_y,test_y),pearsonr(pred_y,test_y)])
joblib.dump(model_svr,"model_svr_new.save")
exit()

#rf
model_rf = RandomForestRegressor(n_estimators=400,min_samples_leaf=4,max_features=250)
model_rf.fit(train_x, train_y)
pred_y = model_rf.predict(test_x)
test_result.append([model_rf.score(test_x,test_y),spearmanr(pred_y,test_y),pearsonr(pred_y,test_y)])


#xgboost
model_xgb = XGBRegressor(n_estimators=150,min_child_weight=32,max_depth=7)
model_xgb.fit(train_x,train_y)
pred_y = model_xgb.predict(test_x)
test_result.append([model_xgb.score(test_x,test_y),spearmanr(pred_y,test_y),pearsonr(pred_y,test_y)])


#decision tree
model_dt = DecisionTreeRegressor(min_samples_leaf=4)
model_dt.fit(train_x,train_y)
pred_y = model_dt.predict(test_x)
test_result.append([model_dt.score(test_x,test_y),spearmanr(pred_y,test_y),pearsonr(pred_y,test_y)])
#logistic reg
# model_log = LogisticRegression()
# model_log.fit(train_x,train_y)
# print(model_log.score(test_x,test_y))

#linear reg
model_lin = LinearRegression()
model_lin.fit(train_x,train_y)
pred_y = model_lin.predict(test_x)
test_result.append([model_lin.score(test_x,test_y),spearmanr(pred_y,test_y),pearsonr(pred_y,test_y)])

#Ridge
model_rig = Ridge()
model_rig.fit(train_x,train_y)
pred_y = model_rig.predict(test_x)
test_result.append([model_rig.score(test_x,test_y),spearmanr(pred_y,test_y),pearsonr(pred_y,test_y)])


#Lasso
model_las = Lasso()
model_las.fit(train_x,train_y)
pred_y = model_las.predict(test_x)
test_result.append([model_las.score(test_x,test_y),spearmanr(pred_y,test_y),pearsonr(pred_y,test_y)])

#KNN
model_knn = KNeighborsRegressor(n_neighbors=20)
model_knn.fit(train_x,train_y)
pred_y = model_knn.predict(test_x)
test_result.append([model_knn.score(test_x,test_y),spearmanr(pred_y,test_y),pearsonr(pred_y,test_y)])

print(name,test_result)
reg_result = [name,test_result]
np.save("reg_result.npy",reg_result)