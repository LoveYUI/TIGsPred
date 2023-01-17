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

con_matrix = []
test_x = np.load("../test_x_new_reg.npy")
test_y = np.load("../test_y_new_reg.npy")
train_x = np.load("../train_x_new_reg.npy")
train_y = np.load("../train_y_new_reg.npy")
pred_y = np.load("pred_y_insr.npy")

for i in range(2,99):
    test_y_c = test_y.copy()
    pred_y_c = pred_y.copy()
    threshold = np.quantile(test_y_c,i/100)
    for j in range(len(test_y_c)):
        if test_y_c[j]>threshold:
            test_y_c[j]=1
        else:
            test_y_c[j]=0
        if pred_y_c[j]>threshold:
            pred_y_c[j]=1
        else:
            pred_y_c[j]=0
    con_matrix.append(confusion_matrix(test_y_c,pred_y_c))

accs = []
precisions = []
recalls = []
for cm in con_matrix:
    tp = cm[1,1]
    fp = cm[0,1]
    tn = cm[0,0]
    fn = cm[1,0]
    accs.append((tp+tn)/(tp+tn+fp+fn))
    precisions.append(tp/(tp+fp))
    recalls.append(tp/(tp+fn))
plt.plot([(i+2)/100 for i in range(len(accs))],accs,color='r')
plt.plot([(i+2)/100 for i in range(len(accs))],precisions,color='g')
plt.plot([(i+2)/100 for i in range(len(accs))],recalls,color='b')
print(accs[47],accs[77],precisions[47],precisions[77],recalls[47],recalls[77])
plt.vlines(0.5, 0, 1, colors = "c", linestyles = "dashed")
plt.vlines(0.75, 0, 1, colors = "lightpink", linestyles = "dashed")
plt.annotate(str(round(accs[48],3)),(0.5,accs[48]+0.01),fontsize=15)
plt.annotate(str(round(precisions[48],3)),(0.5,precisions[48]+0.01),fontsize=15)
plt.annotate(str(round(recalls[48],3)),(0.5,recalls[48]+0.01),fontsize=15)
plt.annotate(str(round(accs[73],3)),(0.75,accs[73]+0.01),fontsize=15)
plt.annotate(str(round(precisions[73],3)),(0.75,precisions[73]+0.01),fontsize=15)
plt.annotate(str(round(recalls[73],3)),(0.75,recalls[73]+0.01),fontsize=15)
plt.gca().set_ylabel('Score',color='k',fontsize=9)
plt.gca().set_xlabel('Threshold(quantiles)',color='k',fontsize=9)

plt.legend(["Accuracy","Precision","Recall"],fontsize=20)

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