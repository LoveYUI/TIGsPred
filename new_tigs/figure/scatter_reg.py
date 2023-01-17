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
import scipy.stats
test_x = np.load("../test_x_new_reg.npy")
test_y = np.load("../test_y_new_reg.npy")
train_x = np.load("../train_x_new_reg.npy")
train_y = np.load("../train_y_new_reg.npy")



#svr
model_svr = SVR(C=3, gamma=0.03125, kernel='rbf')
model_svr.fit(train_x, train_y)
pred_y = model_svr.predict(test_x)
np.save("pred_y_insr.npy",pred_y)
pred_y = np.load("pred_y_insr.npy")
print(scipy.stats.pearsonr(pred_y,test_y))

rank = np.argsort(test_y)
true_y_r = test_y[rank]
pred_y_r = pred_y[rank]
mean = []
box = [[],[],[],[]]
for i in range(len(pred_y_r)):
    if i%10==0:
        mean.append(np.mean(pred_y_r[i-10:i]))

# plt.subplot(221)
for i in range(len(pred_y_r)):
    if test_y[i]>=np.quantile(test_y,0.75):
        box[3].append(pred_y[i])
    elif test_y[i]>=np.quantile(test_y,0.5):
        box[2].append(pred_y[i])
    elif test_y[i]>=np.quantile(test_y,0.25):
        box[1].append(pred_y[i])
    else:
        box[0].append(pred_y[i])

print(np.median(box[0]))
fig=plt.figure(dpi=300)
plt.title("Distribution of predicted TIGs")
plt.suptitle("Distribution of predicted TIGs")

#figure11
ax1=plt.subplot(221)
ax1.scatter([i/(len(true_y_r)) for i in range(len(true_y_r))],pred_y_r,color='b',alpha=0.1,s=2)
ax1.scatter([i/(len(true_y_r)) for i in range(len(true_y_r))],true_y_r,color='r',alpha=0.3,s=4)
ax1.scatter([i/(len(mean)) for i in range(len(mean))],mean,color='lightgreen',alpha=1,s=4)
ax1.set_xlabel('Quantiles of TIGs',color='k',fontsize=8)
ax1.set_ylabel('TIGs value',color='k',fontsize=8)
plt.yticks(fontproperties = 'Times New Roman', size = 6)
plt.xticks(fontproperties = 'Times New Roman', size = 6)
ax1.legend(["Predicted TIGs","Means of Predicted TIGs","True TIGs"])
leg = ax1.get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=5)

# figure12
ax2=plt.subplot(222)
ax2.set_xlabel('Quantiles of TIGs',color='k',fontsize=8)
ax2.set_ylabel('logarithmic TIGs value',color='k',fontsize=8)

ax2.scatter([i/(len(true_y_r)) for i in range(len(true_y_r))],np.log(pred_y_r),color='b',alpha=0.1,s=2)
ax2.scatter([i/(len(true_y_r)) for i in range(len(true_y_r))],np.log(true_y_r),color='r',alpha=0.3,s=4)
ax2.scatter([i/(len(mean)) for i in range(len(mean))],np.log(mean),color='lightgreen',alpha=1,s=4)
ax2.legend(["Predicted TIGs(log)","Means of Predicted TIGs(log)","True TIGs(log)"],fontsize=18)
leg = ax2.get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=5)
plt.yticks(fontproperties = 'Times New Roman', size = 6)
plt.xticks(fontproperties = 'Times New Roman', size = 6)
# figure 22
ax3 = plt.subplot(2,1,2)
# plt.title('Receiver operating characteristic',fontsize = 11)
p = ax3.boxplot(box,sym='',patch_artist=True,vert = False,meanline=True,whiskerprops={'color':'black','alpha':0.2},boxprops={'alpha':1},capprops={'alpha':0.1})
ax3.set_xlabel('Predicted TIGs',color='k',fontsize=9)
ax3.set_ylabel('Intervals between \n quantiles of TIGs',color='k',fontsize=9)
colors = [ 'lightcoral','mediumseagreen','goldenrod','lightskyblue','darkorchid']
colors.reverse()
plt.yticks(fontproperties = 'Times New Roman', size = 6)
plt.xticks(fontproperties = 'Times New Roman', size = 6)
for patch ,color in zip(p['boxes'], colors):
    patch.set_facecolor(color)

plt.tight_layout(pad = 1.5)
plt.show()

