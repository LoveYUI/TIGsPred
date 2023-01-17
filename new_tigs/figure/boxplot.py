import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../../new_tcga_data.csv")
df["Cancer Type"]=df["cancer.type"].apply(lambda x: x.upper())
print(df["new TIGs"]<np.quantile(df["new TIGs"],0.25))
print(np.median(df.loc[df["new TIGs"]<np.quantile(df["new TIGs"],0.25),"new TIGs"]))

#TMB
print(df.columns)
print(df["TMB"].astype("int").max())
df["log(TMB)"]=(df["TMB"].astype("int")/38).apply(np.log)
print(df["Cancer Type"],df["log(TMB)"])
sns.boxplot(x=df["Cancer Type"], y=df["log(TMB)"],order=df.groupby("Cancer Type").median().sort_values(by='TMB').index.values)
plt.xticks(fontsize=16, rotation=30)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("log(TMB)",font2)
plt.xlabel("Cancer Type",fontdict=font2)
plt.subplots_adjust(bottom=0.2)
plt.show()

#APS
sns.boxplot(x=df["Cancer Type"], y=df["new.APM"],order=df.groupby("Cancer Type").median().sort_values(by='new.APM').index.values)
plt.xticks(fontsize=16, rotation=30)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("APS",font2)
plt.xlabel("Cancer Type",fontdict=font2)
plt.subplots_adjust(bottom=0.2)
plt.show()

#TIGs
sns.boxplot(x=df["Cancer Type"], y=df["new TIGs"],order=df.groupby("Cancer Type").median().sort_values(by='new TIGs').index.values)
plt.xticks(fontsize=16, rotation=30)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("TIGs",font2)
plt.xlabel("Cancer Type",fontdict=font2)
plt.subplots_adjust(bottom=0.2)
plt.show()

print(min(df["new TIGs"]),max(df["new TIGs"]))