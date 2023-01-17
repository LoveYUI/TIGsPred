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
from sklearn.metrics import confusion_matrix
from scipy.interpolate import make_interp_spline
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler

test_x = np.load("../test_x_new_reg.npy")
test_y = np.load("../test_y_new_reg.npy")
train_x = np.load("../train_x_new_reg.npy")
train_y = np.load("../train_y_new_reg.npy")

# con_matrix = []
# for i in range(8,93):
#     train_y_c = train_y.copy()
#     test_y_c = test_y.copy()
#     mid = np.quantile(train_y_c,i/100)
#     for j in range(len(train_y_c)):
#         if train_y_c[j] > mid:
#             train_y_c[j] = 1
#         else:
#             train_y_c[j] = 0
#     for j in range(len(test_y_c)):
#         if test_y_c[j] > mid:
#             test_y_c[j] = 1
#         else:
#             test_y_c[j] = 0
#     model_svc = SVC(C=3, gamma=0.08838834764831845, kernel='rbf')
#     model_svc.fit(train_x,train_y_c)
#     con_matrix.append(confusion_matrix(test_y_c,model_svc.predict(test_x)))
#     print(confusion_matrix(test_y_c,model_svc.predict(test_x)))
# np.save("con_matrix.npy",np.array(con_matrix))

con_matrix = np.load("con_matrix.npy")
accs = []
precisions = []
recalls = []
for mat in con_matrix:
    tn = mat[0,0]
    fn = mat[1,0]
    tp = mat[1,1]
    fp = mat[0,1]
    acc = (tp+tn)/(tp+tn+fp+fn)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    accs.append(acc)
    precisions.append(precision)
    recalls.append(recall)
# x_smooth = np.linspace(0,len(accs), 300)
# accs = make_interp_spline([i for i in range(len(accs))], accs)(x_smooth)
print(len(accs),len(precisions),len(recalls))
plt.plot([(i+8)/100 for i in range(len(accs))],accs,color='r',label = "Accuracy")
plt.plot([(i+8)/100 for i in range(len(accs))],precisions,color='g',label = "Precision")
plt.plot([(i+8)/100 for i in range(len(accs))],recalls,color='b',label = "Recall")
plt.legend(fontsize=20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("Score",font2)
plt.xlabel("Threshold(quantiles)",fontdict=font2)
plt.subplots_adjust(bottom=0.2)
plt.show()