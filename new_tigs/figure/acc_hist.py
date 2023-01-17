import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
def score(true_y,pred_y):
    print(confusion_matrix(true_y,pred_y))
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

# cls-acc
# data = [0.878,0.858,0.855,0.854,0.849,0.834,0.798]
#
# label = ["SVM"+"\n"+"rbf","XGboost","Logistic"+"\n"+"Regression","KNN","RF","SVM-"+'\n'+"linear","decision"+"\n"+"tree"]
# ax = plt.gca()
# ax.set_ylim([0.78,0.89])
# ax.set_xlim([-1,7])
# plt.gcf().subplots_adjust(bottom=0.21)
# for x,y in enumerate(data):
#     ax.text(x-0.3,y+0.003,y,va='center',fontsize=10)
# ax.set_xlabel('Model',color='k',fontsize=11)
# ax.set_ylabel('Accuracy',color='k',fontsize=11)
# p = plt.bar(["SVM"+"\n"+"-rbf","XGboost","Logistic"+"\n"+"Regression","KNN","RF","SVM"+'\n'+"-linear","Decision"+"\n"+"tree"],data,width=0.55,label=label,ec='black',color=["coral","orange","gold","lightpink","cyan","greenyellow","slateblue"])
# print(type(p))
# plt.show()



#reg-acc
# data = [0.695,0.597,0.594,0.59,0.5880,0.5814,0.556]
# ax = plt.gca()
# ax.set_ylim([0.53,0.71])
# ax.set_xlim([-1,7])
# plt.gcf().subplots_adjust(bottom=0.21)
# for x,y in enumerate(data):
#     print(x,y)
#     ax.text(x-0.3,y+0.003,y,va='center',fontsize=10)
# ax.set_xlabel('Model',color='k',fontsize=11)
# ax.set_ylabel('R Square',color='k',fontsize=11)
# data.sort(reverse=True)
# plt.bar(["SVR"+"\n"+"-rbf","RF","SVR"+"\n"+"-linear","XGboost","KNN","Ridge","Linear"], data,width=0.55,ec='black',color=["coral","orange","gold","lightpink","cyan","greenyellow","slateblue"])
# # for i in range(len(data)):
# #      plt.text(data[i], data[i], data[i], ha='center')
# plt.show()

#acc
cls_result = np.load("cls_result.npy",allow_pickle=True)
print(cls_result[1][0]["acc"])
true_y = np.load("../test_y_new.npy")
name = ['svr','rf','xgboost','dt','log','lin','knn']
data_acc = []
for i in range(len(name)):
    data_acc.append(cls_result[1][i]["acc"])

name = ['SVM-RBF','RF','XGboost','DecisionTree','Logistic','SVM-Linear','KNN']
data_precision = []
for i in range(len(name)):
    data_precision.append(cls_result[1][i]["precision"])

name = ['SVM-RBF','RF','XGboost','DecisionTree','Logistic','SVM-Linear','KNN']
data_recall = []
for i in range(len(name)):
    data_recall.append(cls_result[1][i]["recall"])

df = pd.DataFrame()
df["name"] = name
df["acc"] = data_acc
df["precision"] = data_precision
df["recall"] = data_recall
print()
x = range(len(list(df.sort_values("acc", ascending=False, inplace=False)["name"])))
plt.figure(figsize=(20,8))
ax = plt.gca()
ax.set_ylim([0.7,0.89])
ax.set_xlim([-1,7])
ax.set_xlabel('Model',color='k',fontsize=11)
ax.set_ylabel('Score',color='k',fontsize=11)
plt.bar([i-0.2 for i in x], df.sort_values("acc", ascending=False, inplace=False)["acc"],width=0.2,ec='black',label = "Accuracy")
plt.bar(x, df.sort_values("acc", ascending=False, inplace=False)["precision"],width=0.2,ec='black',label = "Precision")
plt.bar([i+0.2 for i in x], df.sort_values("acc", ascending=False, inplace=False)["recall"],width=0.2,ec='black',label = "Recall")
for i,j in enumerate(list(df.sort_values("acc", ascending=False, inplace=False)["acc"])):
    print(i,j)
    ax.text(i-0.3,j+0.003,str(round(j,3)),va='center',fontsize=10)
for i,j in enumerate(list(df.sort_values("acc", ascending=False, inplace=False)["precision"])):
    print(i,j)
    ax.text(i-0.1,j+0.003,str(round(j,3)),va='center',fontsize=10)
for i,j in enumerate(list(df.sort_values("acc", ascending=False, inplace=False)["recall"])):
    print(i,j)
    ax.text(i+0.1,j+0.003,str(round(j,3)),va='center',fontsize=10)

plt.xticks(x,df.sort_values("acc", ascending=False, inplace=False)["name"])
plt.legend()
plt.show()
#precision