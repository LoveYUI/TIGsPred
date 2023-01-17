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
def score(true_y,pred_y):
    assert true_y.shape[0]==pred_y.shape[0]
    tn,tp,fn,fp = 0,0,0,0
    for i in range(true_y.shape[0]):
        if true_y[i] >= 0.5:
            if pred_y[i] >= 1:
                tp+=1
            else:
                fn+=1
        else:
            if pred_y[i] < 0.5:
                tn+=1
            else:
                fp+=1
    print(tn,tp,fn,fp)
    score_dict = {"acc":(tp+tn)/(tp+tn+fp+fn),"precision":tp/(tp+fp),"recall":tp/(tp+fn)}
    return score_dict

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
name = ['svr','rf','xgboost','dt','log','lin','knn']
test_result = []

#svr
model_svc = SVC(C=3,gamma=0.015625, kernel='rbf',probability=True)
model_svc.fit(train_x, train_y)
pp = model_svc.predict(test_x)
test_result.append(score(test_y,model_svc.predict(test_x)))
print(classification_report(test_y,model_svc.predict(test_x)))
joblib.dump(model_svc,"model_svc_3_4.save")
exit()
#rf
model_rfcls = RandomForestClassifier(n_estimators=400,min_samples_leaf=2,max_features=320)
model_rfcls.fit(train_x, train_y)
test_result.append(score(test_y,model_rfcls.predict(test_x)))

#xgboost
model_xgb = XGBClassifier(n_estimators=150,min_child_weight=16,max_depth=6)
model_xgb.fit(train_x,train_y)
test_result.append(score(test_y,model_xgb.predict(test_x)))


#decision tree
model_dt = DecisionTreeClassifier(min_samples_leaf=2)
model_dt.fit(train_x,train_y)
test_result.append(score(test_y,model_dt.predict(test_x)))

#logistic reg
model_log = LogisticRegression()
model_log.fit(train_x,train_y)
test_result.append(score(test_y,model_log.predict(test_x)))

#linear reg
model_svc_linear = SVC(kernel='linear')
model_svc_linear.fit(train_x, train_y)
test_result.append(score(test_y,model_svc_linear.predict(test_x)))

#KNN
model_knn = KNeighborsClassifier(n_neighbors=20)
model_knn.fit(train_x,train_y)
test_result.append(score(test_y,model_knn.predict(test_x)))

cls_result = [name,test_result]
print(name,test_result)
np.save("cls_result75.npy",cls_result)