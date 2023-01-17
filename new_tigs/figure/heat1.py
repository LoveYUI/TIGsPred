import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel,ttest_ind

df = pd.read_csv("../../new_tcga_data.csv")
ret = []
name = []
for group in df.groupby("cancer.type"):
    name.append(group[0].upper())
    li = []
    for gene in df.columns[-25:-7]:
        high = group[1].loc[group[1]["TMB"]>=np.median(group[1]["TMB"]),gene]
        low = group[1].loc[group[1]["TMB"]<np.median(group[1]["TMB"]),gene]
        ts = ttest_ind(high,low)
        if ts[1]>0.05:
            li.append(0)
        else:
            li.append(ts[0])
    ret.append(li)

df = pd.DataFrame(ret)
df.columns = ["PSMB5","PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10", "TAP1", "TAP2", "ERAP1", "ERAP2", "CANX", "CALR", "PDIA3", "TAPBP", "B2M", "HLA-A", "HLA-B", "HLA-C"]
df.index = name
print(df.columns)
sns.heatmap(df.T,annot=False, fmt="f",cmap = 'vlag', linewidths=.5,square=True,vmin=-6,vmax=6)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("APM Genes",font2)
plt.xlabel("Cancer Type",fontdict=font2)
plt.subplots_adjust(bottom=0.2)
plt.show()