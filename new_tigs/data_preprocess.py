import numpy as np
from utils import no_quotation,train_and_test
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.svm import SVR,SVC
from unbalence import down_sample
from utils import ml_metric
from imblearn.over_sampling import RandomOverSampler,SMOTE,BorderlineSMOTE,ADASYN,SVMSMOTE,SMOTENC
import imblearn
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler
import collections

#load data csv
data_raw = np.loadtxt("../new_tcga_data.csv",dtype='str',delimiter=',',encoding='utf-8',skiprows=1)
data_raw = data_raw[:,5:]
data_raw = no_quotation(data_raw)
data_raw = data_raw.astype(np.float)

#sep train and test
data_train ,data_test = train_and_test(data_raw,0.8,61)
train_x = data_train[:,:-7]
train_y = data_train[:,-1]
test_x = data_test[:,:-7]
test_y = data_test[:,-1]
print(train_x.shape)

#degs
# degs = np.loadtxt("../DEGs.txt",dtype="str",encoding='utf-8')
# for i in range(degs.shape[0]):
#     degs[i] = degs[i][1:]
# degs = degs.astype(dtype=np.int32)
# train_x = train_x[:,degs-1]
# test_x = test_x[:,degs-1]
degs = np.loadtxt("../new_deg.txt",dtype="str",encoding='utf-8')
degs = degs.astype(dtype=np.int32)
print(degs)
train_x = train_x[:,degs-1]
test_x = test_x[:,degs-1]

#norm
train_x = np.log(train_x+1)
test_x = np.log(test_x+1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_x)
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)
print(train_x.shape,test_x.shape)

#save
model_svr = SVR(C=1.5, gamma=0.022, kernel='rbf')
model_svr.fit(train_x, train_y)
print(model_svr.score(test_x,test_y))
joblib.dump(scaler,"minmaxscaler.save")
np.save("./train_x_new_reg.npy",train_x)
np.save("./train_y_new_reg.npy",train_y)
np.save("./test_x_new_reg.npy",test_x)
np.save("./test_y_new_reg.npy",test_y)

#test
med_tig_count = np.quantile(train_y,0.75)
for i in range(train_y.shape[0]):
    if train_y[i]>med_tig_count:
        train_y[i]=1
    else:
        train_y[i]=0
for i in range(test_y.shape[0]):
    if test_y[i]>med_tig_count:
        test_y[i]=1
    else:
        test_y[i]=0
#
ros = RandomUnderSampler(random_state=0)
train_x, train_y = ros.fit_resample(train_x, train_y)
print(train_x.shape,train_y.shape)
svm_model = SVC(C=3,gamma=0.015625)
svm_model.fit(train_x,train_y)
ml_metric(test_y,svm_model.predict(test_x))
joblib.dump(svm_model,"svm_model_8_25.save")

# print("U")
# ros = RandomUnderSampler(random_state=0)
# train(train_x,train_y,ros)
# print("ClusterCentroids")
# ros = ClusterCentroids()
# train(train_x,train_y,ros)
# print("RandomOverSampler")
# ros = RandomOverSampler(random_state=0)
# train(train_x,train_y,ros)
# ros = SMOTE()
# print("smote")
# train(train_x,train_y,ros)
# ros = BorderlineSMOTE()
# print("bor")
# train(train_x,train_y,ros)
# ros = ADASYN()
# print("ada")
# train(train_x,train_y,ros)
# ros = SVMSMOTE()
# print("svm")
# train(train_x,train_y,ros)
# ros = SMOTENC()
# print("enc")
# train(train_x,train_y,ros)
