import numpy as np
import collections
from utils import no_quotation
import pandas as pd
data_raw = np.loadtxt("../new_tcga_data.csv",dtype='str',delimiter=',',encoding='utf-8',skiprows=1)
data_raw = no_quotation(data_raw)
print(data_raw.shape)
cancer_type = collections.Counter(list(data_raw[:,1]))
med_tigs_all = np.median(data_raw[:,-1].astype(np.float))
uq_tigs_all = np.quantile(data_raw[:,-1].astype(np.float),0.75)
med_tigs = dict()
mean_tigs = dict()
positive_rate = dict()
positive_rate1 = dict()
for ct in cancer_type:
    med_tigs[ct] = np.median(data_raw[np.where(data_raw[:,1]==ct),-1].astype(np.float))
    mean_tigs[ct] = np.mean(data_raw[np.where(data_raw[:, 1] == ct), -1].astype(np.float))
    # print(np.where(data_raw[np.where(data_raw[:, 1] == ct), -1].astype(np.float)>med_tigs_all)[1].shape,data_raw[np.where(data_raw[:, 1] == ct), -1].shape)
    positive_rate[ct] = (np.where(data_raw[np.where(data_raw[:, 1] == ct), -1].astype(np.float)>med_tigs_all)[1].shape[0])/(data_raw[np.where(data_raw[:, 1] == ct), -1].shape[1])
    positive_rate1[ct] = (
                        np.where(data_raw[np.where(data_raw[:, 1] == ct), -1].astype(np.float) > uq_tigs_all)[1].shape[
                            0]) / (data_raw[np.where(data_raw[:, 1] == ct), -1].shape[1])
pd.DataFrame([pd.Series(med_tigs),pd.Series(mean_tigs),pd.Series(positive_rate),pd.Series(positive_rate1)]).to_excel("tabel1.xlsx")