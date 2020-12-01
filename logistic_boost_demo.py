## This is an implementation of the logistic ada boost algorithm

import numpy as np
from sklearn import tree
from data_reader import AbloneDataReader
from logistic_adaboost import LogisticAdaBoost
from adaboost import AdaBoost
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt
# parameters for the training set up
no_training_samples = 3133
no_testing_samples = 1044

reader = AbloneDataReader(no_training_samples, no_testing_samples)

X_train, y_train = reader.get_train_data()
X_test, y_test = reader.get_test_data()
no_base_classifiers = 5
no_folds = 2

# initialising logistic adaboost
f_log = LogisticAdaBoost()

# initialising adaboost
f = AdaBoost()

## CV

mean_acc = []
mean_log_acc = []

for k in range(1, 6):
    no_base_classifiers = k*100
    kf = KFold(n_splits=no_folds)
    cv_acc_arr = []
    cv_log_acc_arr = []
    i = 0
    print("no of base classifiers", no_base_classifiers)
    for train_ind, test_ind in kf.split(X_train):
        # print("cross split no", i)
        x_tr, x_te = X_train.copy()[train_ind], X_train.copy()[test_ind]
        y_tr, y_te = y_train.copy()[train_ind], y_train.copy()[test_ind]
        
        # adaboost
        f.init(x_tr, y_tr)
        f.train(no_base_classifiers)
        y_predict = f.predict(x_te)
        accuracy = np.mean(y_predict == y_te)
        cv_acc_arr.append(accuracy)
        
        # logistic adaboost
        f_log.init(x_tr, y_tr)
        f_log.train(no_base_classifiers)
        y_predict = f_log.predict(x_te)
        accuracy = np.mean(y_predict == y_te)
        cv_log_acc_arr.append(accuracy)
        
        i += 1

    mean_acc.append(np.mean(cv_acc_arr))
    mean_log_acc.append(np.mean(cv_log_acc_arr))


x = 10*np.arange(1,len(mean_acc))
plt.plot(mean_acc, label = "AdaBoost : CV error")
plt.plot(mean_log_acc, label = "Logistic AdaBoost : CV error")
plt.grid()
plt.legend()
plt.show()