import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline
from utils import no_quotation
import seaborn as sns
import pandas as pd
import scipy

data_raw = np.loadtxt("../../new_tcga_data.csv",dtype='str',delimiter=',',encoding='utf-8',skiprows=1)
data_raw = no_quotation(data_raw)
train_y = data_raw[:,100].astype(np.float)
print(pd.read_csv("../../new_tcga_data.csv").columns[100])

print(scipy.stats.normaltest(train_y))
print(scipy.stats.normaltest(np.log(train_y+1)))
sns.kdeplot(train_y/np.max(train_y))

# plt.gca().set_ylim([0,2])
# plt.vlines(np.quantile(train_y,0.5), 0, 2,colors = "r", linestyles = "dashed")
# plt.text(x=900,y=0.004,s=pd.read_csv("../../new_tcga_data.csv").columns[100])


sns.kdeplot(np.log(train_y+1)/np.max(np.log(train_y+1)))
plt.gca().set_xlim([0,1])
plt.xlabel("RNA-Seq")
# plt.gca().set_ylim([0,2])
# plt.vlines(np.quantile(train_y,0.5), 0, 2,colors = "r", linestyles = "dashed")
# plt.text(x=7,y=0.25,s="log(%s)" %pd.read_csv("../../new_tcga_data.csv").columns[100])
plt.show()