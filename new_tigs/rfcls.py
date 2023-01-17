import numpy as np
from utils import *
from sklearn.ensemble import RandomForestClassifier

def cv_train_rfcls(x,y,n_estimators,min_samples_leaf,max_feature,num_cross=10):
    cross_train_x, cross_train_y, cross_valid_x, cross_valid_y = cross_validation(x,y,num_cross)
    num_cross = len(cross_train_x)
    max_score = 0
    score_list=[]
    for i in range(num_cross):
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_feature, n_jobs=-1)
        model.fit(cross_train_x[i], cross_train_y[i])
        score = model.score(cross_valid_x[i], cross_valid_y[i])
        score_list.append(score)
    print(score_list)
    return sum(score_list)/num_cross

def grid_search_rfcls(x,y,n_e_list,min_sl_list,max_f_list,num_cross=10):
    max_ne ,max_msl, max_mf ,max_score= 0,0,0,0
    for i in n_e_list:
        for j in min_sl_list:
            for k in max_f_list:
                score = cv_train_rfcls(x,y,i,j,k,num_cross)
                print("n_estimators",i," min_samples_leaf=",j," max_feature=",k," score=",score,"\n")
                if score>max_score:
                    max_ne = i
                    max_msl = j
                    max_mf = k
                    max_score = score
    model = RandomForestClassifier(n_estimators=max_ne, min_samples_leaf=max_msl,
                                   max_features=max_mf, n_jobs=-1)
    print(max_ne ,max_msl, max_mf ,max_score)
    model.fit(x,y)
    return model

if __name__ == "__main__":
    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")
    mid = np.median(train_y)
    for i in range(len(train_y)):
        if train_y[i] > mid:
            train_y[i] = 1
        else:
            train_y[i] = 0
    num_cross = 10
    model = grid_search_rfcls(train_x,train_y,[100],[1,2,4,8,16,32],[8,16,32,64,128,256],num_cross=10)