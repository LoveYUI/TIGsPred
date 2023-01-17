import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline
from utils import no_quotation
import seaborn as sns
data_raw = np.loadtxt("../../new_tcga_data.csv",dtype='str',delimiter=',',encoding='utf-8',skiprows=1)
data_raw = no_quotation(data_raw)
train_y = data_raw[:,-1].astype(np.float)

sns.distplot(train_y)
# p, x = np.histogram(train_y, bins=train_y.shape[0]//10) # bin it into n = N//10 bins
#
# x = x[:-1] + (x[1] - x[0])/2 # convert bin edges to centers
#
# f = UnivariateSpline(x, p, s=train_y.shape[0])
#
# plt.plot(x, f(x), color='c')
# plt.plot
plt.gca().set_xlim([0,4])
plt.gca().set_ylim([0,2])
plt.vlines(np.quantile(train_y,0.5), 0, 2,colors = "r", linestyles = "dashed")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 30,
}
plt.ylabel("Density",font2)
plt.xlabel("TIGs",fontdict=font2)
plt.subplots_adjust(bottom=0.2)
plt.show()