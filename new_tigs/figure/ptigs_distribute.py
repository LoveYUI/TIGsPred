import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import no_quotation
import joblib
from scipy.stats import spearmanr


df = pd.read_csv("../../new_tcga_data.csv")
df["Cancer Type"]=df["cancer.type"].apply(lambda x: x.upper())

data_raw = np.loadtxt("../../new_tcga_data.csv",dtype='str',delimiter=',',encoding='utf-8',skiprows=1)
data_raw = data_raw[:,5:]
data_raw = no_quotation(data_raw)
data_raw = data_raw.astype(np.float)

#sep train and test
rna_data = data_raw[:,:-7]
rna_data = rna_data.astype(np.float)
rna_data = np.log(rna_data+1)


degs = np.loadtxt("../../new_deg.txt",dtype="str",encoding='utf-8')
degs = degs.astype(dtype=np.int32)
rna_data = rna_data[:,degs-1]
print(rna_data.shape)

scaler = joblib.load("../minmaxscaler.save")
rna_data = scaler.transform(rna_data)

model_svc = joblib.load("./model_svr_new.save")
ptigs = model_svc.predict(rna_data)
print(spearmanr(df["new TIGs"],ptigs))
df["ptigs"] = ptigs

sns.boxplot(x=df["Cancer Type"], y=df["ptigs"],order=df.groupby("Cancer Type").median().sort_values(by='ptigs').index.values)
plt.ylabel("PTIGs")
plt.xticks(fontsize=20, rotation=30)
plt.yticks(fontsize=20)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 24,
}
plt.ylabel("TIGs",font2)
plt.xlabel("Cancer Type",fontdict=font2)
plt.subplots_adjust(bottom=0.2)

plt.show()