import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_theme()

# Load the example flights dataset and convert to long-form
# flights_long = sns.load_dataset("flights")
# flights = flights_long.pivot("month", "year", "passengers")
heat = np.load("heat.npy")

print(heat)
heat_df = pd.DataFrame(heat)
heat_df.columns = ["PSMB5","PSMB6", "PSMB7", "PSMB8", "PSMB9", "PSMB10", "TAP1", "TAP2", "ERAP1", "ERAP2", "CANX", "CALR", "PDIA3", "TAPBP", "B2M", "HLA-A", "HLA-B", "HLA-C","APM","TIGs"]
heat_df.index = ["Spearman's rank correlation coefficient","Kolmogorov-Smirnov test","Mann-Whitney U test","Linregress","Differential Gene"]
heat_df.to_excel("heat1.xlsx")
# Draw a heatmap with the numeric values in each cell
(ax1, ax2) = plt.subplots(figsize=(28,12),nrows=1)
print(ax1)
ax = sns.heatmap(heat_df,annot=False, fmt="f", linewidths=.5,cbar_kws={'label': 'Test score','orientation': 'horizontal'},square=True)
plt.gcf().subplots_adjust(left=0.24)

#设置colorbar的刻度字体大小
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=16)
cbar = ax.collections[0].colorbar
cbar.set_label('Test score',fontdict={'family':'Times New Roman','size':18, 'color':'#000000'})

plt.xticks(fontsize=18, rotation=-45)
plt.yticks(fontsize=18)


plt.show()