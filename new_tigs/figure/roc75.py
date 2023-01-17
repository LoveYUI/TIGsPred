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
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics
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

# svr
model_svc = SVC(C=3,gamma=0.015625, kernel='rbf',probability = True)
model_svc.fit(train_x, train_y)
prob_svc_y = model_svc.predict_proba(test_x)
np.save("prob_svc_y75.npy",prob_svc_y)
#rf
model_rfcls = RandomForestClassifier(n_estimators=400,min_samples_leaf=2,max_features=320)
model_rfcls.fit(train_x, train_y)
prob_rf_y = model_rfcls.predict_proba(test_x)
np.save("prob_rf_y75.npy",prob_rf_y)


#xgboost
model_xgb = XGBClassifier(n_estimators=150,min_child_weight=16,max_depth=6)
model_xgb.fit(train_x,train_y)
prob_xgb_y = model_xgb.predict_proba(test_x)
np.save("prob_xgb_y75.npy",prob_xgb_y)


#decision tree
model_dt = DecisionTreeClassifier(min_samples_leaf=2)
model_dt.fit(train_x,train_y)
prob_dt_y = model_dt.predict_proba(test_x)

#logistic reg
model_log = LogisticRegression()
model_log.fit(train_x,train_y)
prob_log_y = model_log.predict_proba(test_x)

#linear reg
model_svc_linear = SVC(kernel='linear',probability=True)
model_svc_linear.fit(train_x, train_y)
prob_svcl_y = model_svc_linear.predict_proba(test_x)

#KNN
model_knn = KNeighborsClassifier(n_neighbors=20)
model_knn.fit(train_x,train_y)
prob_knn_y = model_knn.predict_proba(test_x)

prob_svc_y = np.load("prob_svc_y75.npy")
prob_rf_y = np.load("prob_rf_y75.npy")
prob_xgb_y = np.load("prob_xgb_y75.npy")
fpr_svc, tpr_svc, thersholds_svc = roc_curve(test_y, prob_svc_y[:,1])
print(metrics.roc_auc_score(test_y,prob_svc_y[:,1]))
fpr_rf, tpr_rf, thersholds_rf = roc_curve(test_y, prob_rf_y[:,1])
print(metrics.roc_auc_score(test_y,prob_rf_y[:,1]))
fpr_xgb, tpr_xgb, thersholds_xgb = roc_curve(test_y, prob_xgb_y[:,1])
print(metrics.roc_auc_score(test_y,prob_xgb_y[:,1]))
# fpr_dt, tpr_dt, thersholds_dt = roc_curve(test_y, prob_dt_y[:,1])
print(metrics.roc_auc_score(test_y,prob_dt_y[:,1]))
fpr_log, tpr_log, thersholds_log = roc_curve(test_y, prob_log_y[:,1])
print(metrics.roc_auc_score(test_y,prob_log_y[:,1]))
fpr_svcl, tpr_svcl, thersholds_svcl = roc_curve(test_y, prob_svcl_y[:,1])
print(metrics.roc_auc_score(test_y,prob_svcl_y[:,1]))
fpr_knn, tpr_knn, thersholds_knn = roc_curve(test_y, prob_knn_y[:,1])
print(metrics.roc_auc_score(test_y,prob_knn_y[:,1]))
plt.plot(fpr_svc,tpr_svc,label="SVM-rbf:"+'      '+str(round(metrics.roc_auc_score(test_y,prob_svc_y[:,1]),3)))

plt.plot(fpr_xgb,tpr_xgb,label="XGBoost:"+'     '+str(round(metrics.roc_auc_score(test_y,prob_xgb_y[:,1]),3)))
plt.plot(fpr_rf,tpr_rf,label="RF:"+'               '+str(round(metrics.roc_auc_score(test_y,prob_rf_y[:,1]),3)))
# plt.plot(fpr_dt,tpr_dt,label="Decision Tree")
plt.plot(fpr_log,tpr_log,label="Logistic:"+'       '+str(round(metrics.roc_auc_score(test_y,prob_log_y[:,1]),3)))
plt.plot(fpr_svcl,tpr_svcl,label="SVM-linear"+'   '+str(round(metrics.roc_auc_score(test_y,prob_svcl_y[:,1]),3)))
plt.plot(fpr_knn,tpr_knn,label="KNN:"+'            '+str(round(metrics.roc_auc_score(test_y,prob_knn_y[:,1]),3)))



ax = plt.gca()
plt.title('Receiver operating characteristic',fontsize = 24)
plt.legend(loc="best")

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("TP Rate",font2)
plt.xlabel("FP Rate",fontdict=font2)
plt.subplots_adjust(bottom=0.2)

plt.show()

