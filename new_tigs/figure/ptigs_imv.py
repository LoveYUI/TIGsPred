import scipy
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
import pandas as pd
import seaborn as sns

imv_pData = pd.read_csv("D:/work/毕设/论文修改20211230/pData.csv")
ptigs = np.loadtxt("D:/work/毕设/论文修改20211230/IMvigor210_ptigs.csv",skiprows=0,delimiter=',',dtype="str")[:,-1]
imv_pData["PTIGs"] = ptigs.astype("float")
print(imv_pData.columns)
sns.boxplot(x=imv_pData["binaryResponse"], y=imv_pData["PTIGs"])
group = []
for group1 in imv_pData.groupby("binaryResponse"):
    group.append(group1)

plt.text(x=-0.35,y=1.05,s="p-value=%s"%round(stats.ttest_ind(group[0][1]["PTIGs"],group[1][1]["PTIGs"])[1],4),fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("PTIGs",font2)
plt.xlabel("binaryResponse",fontdict=font2)
plt.show()

group = []
imv_pData = imv_pData[imv_pData["Best Confirmed Overall Response"]!="NE"]
sns.boxplot(x=imv_pData["Best Confirmed Overall Response"], y=imv_pData["PTIGs"],order=imv_pData.groupby("Best Confirmed Overall Response").median().sort_values(by='PTIGs').index.values,fliersize=0)
plt.hlines(y =0.85, xmin =0.1, xmax =1.9,color='k',linestyles='--')
plt.hlines(y =1.0, xmin =0.1, xmax =2.9,color='k',linestyles='--')
plt.hlines(y =0.3, xmin =1.1, xmax =2.9,color='k',linestyles='--')
for group1 in imv_pData.groupby("Best Confirmed Overall Response"):
    group.append(group1)
    print(group1[0])
plt.text(x= 0.7,y=0.86,s="p-value=%s"%round(stats.ttest_ind(group[1][1]["PTIGs"],group[2][1]["PTIGs"])[1],4),fontsize=18)
plt.text(x=1.2,y=1.01,s="p-value=%s"%round(stats.ttest_ind(group[3][1]["PTIGs"],group[0][1]["PTIGs"])[1],4),fontsize=18)
plt.text(x=1.7,y=0.31,s="p-value=%s"%round(stats.ttest_ind(group[3][1]["PTIGs"],group[0][1]["PTIGs"])[1],4),fontsize=18)
# print(round(stats.ttest_ind(group[1][1]["PTIGs"],group[3][1]["PTIGs"])[1],4))
# print(round(stats.ttest_ind(group[3][1]["PTIGs"],group[2][1]["PTIGs"])[1],4),fontsize=18)
# print(round(stats.ttest_ind(group[2][1]["PTIGs"],group[0][1]["PTIGs"])[1],4),fontsize=18)
# print(round(stats.ttest_ind(group[1][1]["PTIGs"],group[2][1]["PTIGs"])[1],4),fontsize=18)
# print(round(stats.ttest_ind(group[1][1]["PTIGs"],group[0][1]["PTIGs"])[1],4),fontsize=18)
# print(round(stats.ttest_ind(group[3][1]["PTIGs"],group[0][1]["PTIGs"])[1],4),fontsize=18)
# plt.text(x=-0.35,y=1.05,s="p-value=%s"%round(stats.ttest_ind(group[0][1]["PTIGs"],group[1][1]["PTIGs"])[1],4))

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("PTIGs",font2)
plt.xlabel("Response",fontdict=font2)
plt.show()

apm = pd.read_csv("D:/work/毕设/论文修改20211230/apm_imv.csv").iloc[:,1:].T
imv_pData["apm"] = list(apm[0])
print(imv_pData["apm"])
# print(np.log(1+imv_pData["FMOne mutation burden per MB"]))
# print(imv_pData["apm"])
# print((imv_pData["apm"]-min(imv_pData["apm"]))/max((imv_pData["apm"])-min(imv_pData["apm"])))
imv_pData["tigs"] = np.log10(1+imv_pData["FMOne mutation burden per MB"])*((imv_pData["apm"]-min(imv_pData["apm"]))/(max(imv_pData["apm"])-min(imv_pData["apm"])))
print(imv_pData["tigs"])
print(imv_pData["tigs"].corr(imv_pData["PTIGs"]))

true_y_r = np.array(imv_pData.loc[~pd.isna(imv_pData["tigs"]),"tigs"])
pred_y_r = np.array(imv_pData.loc[~pd.isna(imv_pData["tigs"]),"PTIGs"])
rank = np.argsort(true_y_r)
true_y_r = true_y_r[rank]
pred_y_r = pred_y_r[rank]
plt.scatter([i/(len(true_y_r)) for i in range(len(true_y_r))],pred_y_r,color='b',alpha=0.4,s=2)
plt.scatter([i/(len(true_y_r)) for i in range(len(true_y_r))],true_y_r,color='r',alpha=0.3,s=4)
plt.show()
# def binary(x):
#     if x=='CR/PR':
#         return 1
#     else:
#         return 0
#
# imv_pData["Res01"]=imv_pData["Best Confirmed Overall Response"].apply(binary)
# imv_pData = imv_pData[~pd.isna(imv_pData["Res01"])]
# fpr_pt, tpr_pt, thersholds_pt = roc_curve(imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"Res01"], imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"PTIGs"])
# fpr_tmb, tpr_tmb, thersholds_tmb = roc_curve(imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"Res01"], imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"FMOne mutation burden per MB"])
# plt.plot(fpr_pt,tpr_pt)
# plt.plot(fpr_tmb,tpr_tmb)
# print(metrics.roc_auc_score(imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"Res01"], imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"tigs"]))
# print(metrics.roc_auc_score(imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"Res01"], imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"PTIGs"]))
# print(metrics.roc_auc_score(imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"Res01"], imv_pData.loc[~pd.isna(imv_pData["FMOne mutation burden per MB"]),"FMOne mutation burden per MB"]))
# plt.show()

