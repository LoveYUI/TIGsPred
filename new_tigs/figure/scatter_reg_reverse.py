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
from scipy.stats import ttest_ind
test_x = np.load("../test_x_new_reg.npy")
test_y = np.load("../test_y_new_reg.npy")
train_x = np.load("../train_x_new_reg.npy")
train_y = np.load("../train_y_new_reg.npy")
pred_y = np.load("pred_y_insr.npy")

box = [[],[],[],[]]
for i in range(len(pred_y)):
    if pred_y[i]>=np.quantile(pred_y,0.75):
        box[3].append(test_y[i])
    elif pred_y[i]>=np.quantile(pred_y,0.5):
        box[2].append(test_y[i])
    elif pred_y[i]>=np.quantile(pred_y,0.25):
        box[1].append(test_y[i])
    else:
        box[0].append(test_y[i])

print((np.where(np.array(box[0])<0.36))[0].shape)
print((np.where(np.array(box[1])<0.36))[0].shape)
print(len(box[0]),len(box[1]))
print(ttest_ind(box[0],box[1]),ttest_ind(box[0],box[2]),ttest_ind(box[0],box[3]),ttest_ind(box[1],box[2]),ttest_ind(box[1],box[3]),ttest_ind(box[2],box[3]))

plt.plot([0.5,4.5],[np.quantile(train_y,0.5),np.quantile(train_y,0.5)],color = 'r',alpha = 0.5)
plt.plot([0.5,4.5],[np.quantile(train_y,0.75),np.quantile(train_y,0.75)],color = 'b',alpha = 0.5)
plt.legend(["the median of TIGs","the first quintile of TIGs"],fontsize=20)
p = plt.boxplot(box,sym='+',patch_artist=True,meanline=True,flierprops={"alpha":0.2},whiskerprops={'color':'black','alpha':0.2},boxprops={'alpha':1},capprops={'alpha':0.1})
plt.gca().set_ylabel('True TIGs',color='k',fontsize=9)
plt.gca().set_xlabel('Intervals between \n quantiles of predicted TIGs',color='k',fontsize=9)
colors = [ 'lightcoral','mediumseagreen','goldenrod','darkorchid']
colors.reverse()

plt.yticks(fontproperties = 'Times New Roman', size = 6)
plt.xticks(fontproperties = 'Times New Roman', size = 6)
for patch ,color in zip(p['boxes'], colors):
    patch.set_facecolor(color)


plt.title("Distribution of True TIGs under predicted TIGs",fontdict={'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 30,
})
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 28,
}
plt.ylabel("TIGs",font2)
plt.xlabel('Intervals between \n quintiles of predicted TIGs',fontdict=font2)
plt.subplots_adjust(bottom=0.2)

plt.show()


# 80% for line
probs = []
