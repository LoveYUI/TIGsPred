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
import matplotlib.patches
import matplotlib.lines
import seaborn as sns
import scipy.stats

def linear_regression(x,y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N,sumx],[sumx,sumx2]])
    b = np.array([sumy,sumxy])

    return np.linalg.solve(A,b)
def least_squares(x,y):
    ones = np.ones((len(x)))
    A = np.c_[ones,x]
    ATA = A.T.dot(A)
    ATb = A.T.dot(y)
    inv_ATA = np.linalg.inv(ATA)
    solution = inv_ATA.dot(ATb)
    return solution


test_x = np.load("../test_x_new_reg.npy")
test_y = np.load("../test_y_new_reg.npy")
train_x = np.load("../train_x_new_reg.npy")
train_y = np.load("../train_y_new_reg.npy")
prob_svc_y = np.load("prob_svc_y.npy")[:,1]
med = np.median(test_y)
a0,a1 = least_squares(prob_svc_y,test_y)
y1=[]
y2=[]
x1=[]
x2=[]
for i,tig in enumerate(test_y):
    if tig > med:
        y2.append(tig)
        x2.append(prob_svc_y[i])
    else:
        y1.append(tig)
        x1.append(prob_svc_y[i])
tig = [[],[],[],[],[]]
for i,t in enumerate(prob_svc_y):
    if t > 0.8:
        tig[4].append(test_y[i])
    elif t > 0.6:
        tig[3].append(test_y[i])
    elif t >0.4:
        tig[2].append(test_y[i])
    elif t >0.2:
        tig[1].append(test_y[i])
    else:
        tig[0].append(test_y[i])
# x=[0.1,0.3,0.5,0.7,0.9],y=
# plt.boxplot(tig,positions=[0.1,0.3,0.5,0.7,0.9],widths=0.05,sym='')

print(prob_svc_y.shape)
print(scipy.stats.pearsonr(prob_svc_y,np.log(test_y)))
ax = plt.gca()
ax.set_ylim([-0.01,1.01])
ax.set_xlim([-0.01,1.01])
plt.text(0.3,0.95,"pearson=0.795, p-value<0.001")
plt.scatter(x1,y1,s=10,color='#00CED1',alpha=0.4)
plt.scatter(x2,y2,s=10,color='deeppink',alpha=0.4)
plt.legend(["Negative","Positive"],loc="upper left")
ax.set_xlabel('Probability of TIGs',color='k',fontsize=11)
ax.set_ylabel('TIGs',color='k',fontsize=11)
plt.hlines(np.median(test_y), 0,1,colors="#00CED1",linestyles="--",alpha=0.5)
plt.hlines(np.median(test_y)+0.001, 0,1,colors="deeppink",linestyles="--",alpha=0.5)
plt.plot([0,1],[a0,a0+a1],color='brown',linewidth=1.5)
plt.show()