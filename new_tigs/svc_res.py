import numpy as np
from utils import *
from sklearn.svm import SVC,SVR
from imblearn.over_sampling import SVMSMOTE
def cv_train_svc(x,y,c,gam,num_cross=10):
    cross_train_x, cross_train_y, cross_valid_x, cross_valid_y = cross_validation(x,y,num_cross)
    num_cross = len(cross_train_x)
    max_score = 0
    score_list=[]
    for i in range(num_cross):
        model = SVC(C=c, gamma=gam, kernel='rbf')
        roc = SVMSMOTE()
        cross_train_x_res,cross_train_y_res = roc.fit_resample(cross_train_x[i], cross_train_y[i])
        model.fit(cross_train_x_res,cross_train_y_res)
        score = model.score(cross_valid_x[i], cross_valid_y[i])
        score_list.append(score)
    print(score_list)
    return sum(score_list)/num_cross

def grid_search_svc(x,y,c_list,gam_list,num_cross=10):
    max_c ,max_gam, max_score = 0,0,0
    for i in c_list:
        for j in gam_list:
            score = cv_train_svc(x,y,i,j,num_cross)
            print("c=",i," gam=",j," score=",score,"\n")
            if score>max_score:
                max_c = i
                max_gam = j
                max_score = score
    model = SVC(C=max_c, gamma=max_gam, kernel='rbf')
    print(max_c,max_gam,max_score)
    model.fit(x,y)
    return model

if __name__ == "__main__":
    train_x = np.load("train_x_new_reg.npy")
    train_y = np.load("train_y_new_reg.npy")
    mid = np.quantile(train_y,0.75)
    for i in range(len(train_y)):
        if train_y[i] > mid:
            train_y[i] = 1
        else:
            train_y[i] = 0
    num_cross = 10
    model = grid_search_svc(train_x, train_y, [1,2,3],[2 ** (-1), 2 ** (-2),2 ** (-3),2 ** (-4),2 ** (-5), 2 ** (-6),2 ** (-7),2 ** (-8),2 ** (-9)], num_cross)
