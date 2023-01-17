import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("../../new_tcga_data.csv")
df["Cancer Type"]=df["cancer.type"].apply(lambda x: x.upper())
print(df.TMB.median())
df.loc[df.TMB>df.TMB.median(),"TMB statue"]="High"
df.loc[df.TMB<=df.TMB.median(),"TMB statue"]="Low"
print(df["TMB statue"])
fig, ax = plt.subplots()
sns.boxplot(x=df["Cancer Type"], y=df["new.APM"],hue=df["TMB statue"],order=df.groupby("Cancer Type").median().sort_values(by='new.APM').index.values,width=0.45,fliersize=0,ax=ax)
ax.set_ylim(-1,1.5)
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