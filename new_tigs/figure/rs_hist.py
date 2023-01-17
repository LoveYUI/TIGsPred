import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

reg_result = np.load("reg_result.npy",allow_pickle=True)
print(len(reg_result[1]))
rs,spr,psr = [],[],[]
for i in range(len(reg_result[1])):
    rs.append(reg_result[1][i][0])
for i in range(len(reg_result[1])):
    spr.append(reg_result[1][i][1][0])
for i in range(len(reg_result[1])):
    psr.append(reg_result[1][i][2][0])
rs.pop(-2)
spr.pop(-2)
psr.pop(-2)
name = ['SVM-Linear','SVM-RBF','RF','XGBoost','Decision Tree','Linear Regression','Ridge','knn']

df = pd.DataFrame()
df["name"] = name
df["rs"] = rs
df["spr"] = spr
df["psr"] = psr

x = range(len(list(df.sort_values("rs", ascending=False, inplace=False)["name"])))
plt.figure(figsize=(20,8))
ax = plt.gca()
ax.set_ylim([0,0.9])
ax.set_xlim([-1,8])
ax.set_xlabel('Model',color='k',fontsize=11)
ax.set_ylabel('Score',color='k',fontsize=11)
plt.bar([i-0.2 for i in x], df.sort_values("rs", ascending=False, inplace=False)["rs"],width=0.2,ec='black',label = "R Square")
plt.bar(x, df.sort_values("rs", ascending=False, inplace=False)["spr"],width=0.2,ec='black',label = "Spearman")
plt.bar([i+0.2 for i in x], df.sort_values("rs", ascending=False, inplace=False)["psr"],width=0.2,ec='black',label = "Pearson")
for i,j in enumerate(list(df.sort_values("rs", ascending=False, inplace=False)["rs"])):
    print(i,j)
    ax.text(i-0.3,j+0.003,str(round(j,3)),va='center',fontsize=12)
for i,j in enumerate(list(df.sort_values("rs", ascending=False, inplace=False)["spr"])):
    print(i,j)
    ax.text(i-0.1,j+0.003,str(round(j,3)),va='center',fontsize=12)
for i,j in enumerate(list(df.sort_values("rs", ascending=False, inplace=False)["psr"])):
    print(i,j)
    ax.text(i+0.1,j+0.003,str(round(j,3)),va='center',fontsize=12)

plt.xticks(x,df.sort_values("rs", ascending=False, inplace=False)["name"])
plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("Score",font2)
plt.xlabel("Model",fontdict=font2)

plt.show()