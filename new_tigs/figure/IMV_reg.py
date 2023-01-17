import numpy as np
from sklearn.svm import SVC,SVR
from utils import no_quotation,train_and_test
import joblib
from sklearn.preprocessing import MinMaxScaler
test_y = np.load("../test_y_new_reg.npy")
threshold = np.quantile(test_y,0.75)

#pre
IMvigor210_data = np.loadtxt("D:\\work\\毕设\\论文修改20211230\\IMvigor210_new.csv",dtype='str',delimiter=',',encoding='utf-8',skiprows=1)
IMvigor210_data = IMvigor210_data[:,1:]
IMvigor210_data = no_quotation(IMvigor210_data)
rna_data = IMvigor210_data
rna_data = rna_data.astype(np.float)
rna_data = np.log(rna_data+1)
print(rna_data.shape)
# degs = np.loadtxt("DEGs.txt",dtype="str",encoding='utf-8')
# for i in range(degs.shape[0]):
#     degs[i] = degs[i][1:]
# degs = degs.astype(dtype=np.int32)

degs = np.loadtxt("../../new_deg.txt",dtype="str",encoding='utf-8')
degs = degs.astype(dtype=np.int32)
rna_data = rna_data[:,degs-1]
print(rna_data.shape)

scaler = joblib.load("../minmaxscaler.save")
rna_data = scaler.transform(rna_data)

model_svc = joblib.load("./model_svr_new.save")
ptigs = model_svc.predict(rna_data)
for i in range(ptigs.shape[0]):
    if ptigs[i]>threshold:
        ptigs[i]=1
    else:
        ptigs[i]=0
data_all_new = np.concatenate((IMvigor210_data,ptigs.reshape(ptigs.shape[0],1)),axis=1)
print(ptigs)
print(data_all_new.shape)
np.savetxt("D:\\work\\毕设\\论文修改20211230\\IMvigor210_ptigs_reg.csv",data_all_new,delimiter=',',fmt='%s')