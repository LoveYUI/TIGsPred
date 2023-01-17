import numpy as np
from utils import no_quotation
from sklearn.svm import SVC,SVR
from scipy.stats import ttest_rel,ttest_ind
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t
from scipy.stats import spearmanr,pearsonr,wilcoxon,mannwhitneyu
from scipy.stats import kstest,ks_2samp,linregress
import pandas as pd
df = pd.read_csv("../../new_tcga_data.csv")
df["log(TMB)"]=(df["TMB"].astype("int")/38).apply(np.log)
tmb = np.array(df["log(TMB)"])
tmb.reshape((tmb.shape[0],1))
apm = np.array(df["new.APM"])
apm.reshape((apm.shape[0],1))
tigs = np.array(df["new TIGs"])
tigs.reshape((tigs.shape[0],1))
genes = np.array(df.iloc[:,-25:-7])
print(tmb.shape,apm.shape,tigs.shape,genes.shape)
gene_apm_tigs = np.concatenate((genes,apm.reshape([apm.shape[0],1]),tigs.reshape([tigs.shape[0],1])),axis=1)
median_tmb = np.median(tmb)
gene_apm_tigs_high = gene_apm_tigs[np.where(tmb > median_tmb)[0], :]
gene_apm_tigs_low = gene_apm_tigs[np.where(tmb <= median_tmb)[0], :]

#ks 检验
ks = []
for i in range(gene_apm_tigs.shape[1]):
    ks_ret = ks_2samp(gene_apm_tigs_high[:,i],gene_apm_tigs_low[:,i])
    if ks_ret[1]<0.05:
        ks.append(ks_ret[0])
    else:
        ks.append(0)
ks = np.array(ks)
ks = ks.reshape(1,ks.shape[0])
#
#spearman检验
spr = []
for i in range(gene_apm_tigs.shape[1]):
    sp_ret = spearmanr(gene_apm_tigs_high[:min(gene_apm_tigs_high.shape[0],gene_apm_tigs_low.shape[0]),i],gene_apm_tigs_low[:min(gene_apm_tigs_high.shape[0],gene_apm_tigs_low.shape[0]),i])
    if sp_ret[1] < 0.05:
        spr.append(sp_ret[0])
    else:
        spr.append(0)
spr = np.array(spr)
spr = spr.reshape(1,spr.shape[0])
#
# #wilcoxon检验
# for i in range(gene_apm_tigs.shape[1]):
#     wc_ret = wilcoxon(np.sort(gene_apm_tigs_high[:min(gene_apm_tigs_high.shape[0],gene_apm_tigs_low.shape[0]),i]),np.sort(gene_apm_tigs_low[:min(gene_apm_tigs_high.shape[0],gene_apm_tigs_low.shape[0]),i]))
#     print(wc_ret)
#
# #mannwhitneyu检验
mwu = []
for i in range(gene_apm_tigs.shape[1]):
    mwu_ret = mannwhitneyu(gene_apm_tigs_high[:min(gene_apm_tigs_high.shape[0],gene_apm_tigs_low.shape[0],200),i],gene_apm_tigs_low[:min(gene_apm_tigs_high.shape[0],gene_apm_tigs_low.shape[0],200),i])
    if mwu_ret[1] < 0.05:
        mwu.append(mwu_ret[0])
        # mwu.append(-np.log10(mwu_ret[1]))
    else:
        mwu.append(0)
mwu = np.array(mwu)/np.max(mwu)
mwu = mwu.reshape(1,mwu.shape[0])
#Stats.linregress
lin = []
for i in range(gene_apm_tigs.shape[1]):
    # print(list(gene_apm_tigs[:,i].flatten()),list(tmb))
    lin_ret = linregress(list(gene_apm_tigs[:,i].flatten()),list(tmb.flatten()))
    if lin_ret[3] < 0.05:
        lin.append(lin_ret[2])
    else:
        lin.append(0)
lin = np.array(lin)
lin = lin.reshape(1,lin.shape[0])


deg = [-0.13952054, -0.20235395, -0.22468231,  0.20079217,  0.43271332,  0.10410169,  0.50875813,
0.26399990 ,-0.13704363, -0.18884181, -0.49473828, -0.16845841, -0.26478192,  0.06212003,
-0.24419899,  0.06607185,  0.04248935, -0.02402072,0,0]
deg = np.array(np.abs(deg))
deg = deg.reshape(1,deg.shape[0])
heat = np.concatenate((spr,ks,mwu,lin,deg),axis =0)
np.save("heat.npy",heat)
# model_svr = SVR(kernel='rbf')
# model_svr.fit(genes,apm)
# print(model_svr.score(genes,tmb))

# t-test p-value, person, spearman, regression t-test
# tt = np.concatenate((genes,tmb.reshape(tmb.shape[0],1)),axis=1)
# med = np.median(tt[:,-1])
# for i in range(tt.shape[0]):
#     if tt[i,-1]>med:
#         tt[i,-1]=1
#     else:
#         tt[i,-1]=0
# # print(ttest_ind([-0.78571512, -0.69826573,-0.66055641, 0.31116072, -0.94367846,
# #   -0.96233765],[-0.29325476, -0.05130731 ,-0.88141009,  -0.32023061, -0.96246204
# #   ,-1.0731199 ]))
# # exit()
# for i in range(tt.shape[1]-1):
#     a = tt[np.where(tt[:,-1]==1),i]
#     b = tt[np.where(tt[:, -1] == 0), i]
#     print(ttest_ind(a[0, :2000], b[0,:2000]))
#     # if a.shape[1]>b.shape[1]:
#     #     print(np.isnan(a), np.isnan(b))
#     #     print(ttest_ind(a[0:,b.shape[1]],b[0]))
#     # else:
#     #     print(np.isnan(a),np.isnan(b))
#     #     print(ttest_ind(a[0], b[0:,:a.shape[0]]))

# def KSTest(mat):
#     pass
# psr = np.concatenate((genes,tmb.reshape(tmb.shape[0],1)),axis=1)
# psr_v = []
# psr_p = []
# spr_v = []
# spr_p = []
# for i in range(psr.shape[1]-1):
#     psr_p.append(pearsonr(psr[:,i],psr[:,-1])[1])
#     spr_p.append(spearmanr(psr[:,i],psr[:,-1])[1])
#     psr_v.append(pearsonr(psr[:, i], psr[:, -1])[0])
#     spr_v.append(spearmanr(psr[:, i], psr[:, -1])[0])
# print(spr_p)