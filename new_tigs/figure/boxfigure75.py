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
from scipy import stats

test_x = np.load("../test_x_new_reg.npy")
test_y = np.load("../test_y_new_reg.npy")
train_x = np.load("../train_x_new_reg.npy")
train_y = np.load("../train_y_new_reg.npy")
prob_svc_y = np.load("prob_svc_y75.npy")[:,1]
print(prob_svc_y.shape)

tigs = [[],[]]
ptigs = [[],[]]
for i,prob in enumerate(prob_svc_y):
    if prob>0.5:
        tigs[1].append(test_y[i])
    else:
        tigs[0].append(test_y[i])
for i in range(len(test_y)):
    if test_y[i]>np.median(test_y):
        ptigs[1].append(prob_svc_y[i])
    else:
        ptigs[0].append(prob_svc_y[i])


p = plt.boxplot(tigs,sym='',patch_artist=True,widths=0.4,showmeans=True,meanline=True,whiskerprops={'color':'black','alpha':0.5},boxprops={'alpha':0.8},capprops={'alpha':0.5})

print(p)
colors = [ 'lightblue','pink']
for patch ,color in zip(p['boxes'], colors):
    patch.set_facecolor(color)
    # patch.set_edgecolor(color)
for med in p['medians']:
    med.set_color('red')
for mean in p['means']:
    mean.set_color('green')

#添加网格线
ax = plt.gca()
ax.yaxis.grid(True, color='black', linestyle='--', linewidth=1,alpha=0.15) #在y轴上添加网格线
ax.set_xticks([y+1 for y in range(len(tigs))] ) #指定x轴的轴刻度个数
plt.setp(ax, xticks=[1,2],
         xticklabels=['Low TIGs\n(Predicted)','High TIGs\n(Predicted)'])

## [y+1 for y in range(len(all_data))]运行结果是[1,2,3]
plt.gcf().subplots_adjust(bottom=0.21)
plt.text(0.65, 2.11, "p-value<0.001")
print(stats.ttest_ind(tigs[0], tigs[1], equal_var=False))

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 40,
}
plt.ylabel('TIGs',font2)
plt.xlabel('Predicted TIGs',fontdict=font2)

plt.show()

p = plt.boxplot(ptigs,sym='',patch_artist=True,widths=0.4,showmeans=True,meanline=True,whiskerprops={'color':'black','alpha':0.5},boxprops={'alpha':0.8},capprops={'alpha':0.5})

print(p)
colors = [ 'lightblue','pink']
for patch ,color in zip(p['boxes'], colors):
    patch.set_facecolor(color)
    # patch.set_edgecolor(color)
for med in p['medians']:
    med.set_color('red')
for mean in p['means']:
    mean.set_color('green')

#添加网格线
ax = plt.gca()
ax.yaxis.grid(True, color='black', linestyle='--', linewidth=1,alpha=0.15) #在y轴上添加网格线
ax.set_xticks([y+1 for y in range(len(tigs))] ) #指定x轴的轴刻度个数
plt.setp(ax, xticks=[1,2],
         xticklabels=['Low TIGs','High TIGs'])

## [y+1 for y in range(len(all_data))]运行结果是[1,2,3]
plt.gcf().subplots_adjust(bottom=0.21)
# plt.text(0.65, 1, "p-value<0.001")
print(stats.ttest_ind(tigs[0], tigs[1], equal_var=False))
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 40,
}
plt.ylabel('Predicted TIGs probability',font2)
plt.xlabel('TIGs',fontdict=font2)
plt.show()