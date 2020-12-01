## This is an implementation of the ada boost algorithm
import numpy as np
from sklearn import tree
from data_reader import AbloneDataReader

from adaboost import AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold


# parameters for the training set up
no_training_samples = 3133
no_testing_samples = 1044

reader = AbloneDataReader(no_training_samples, no_testing_samples)

X_train, y_train = reader.get_train_data()
X_test, y_test = reader.get_test_data()
no_base_classifiers = 200
no_folds = 2
# initialising adaboost
f = AdaBoost()

## sklearn implementation of boost
boost = AdaBoostClassifier( base_estimator = tree.DecisionTreeClassifier(max_depth = 1, max_leaf_nodes=2), 
                            algorithm = 'SAMME',n_estimators=no_base_classifiers, learning_rate=1.0)

## CV
kf = KFold(n_splits=no_folds)
cv_acc_arr = []
cv_sk_acc_arr = []
i = 0
for train_ind, test_ind in kf.split(X_train):
    print("cross split no", i)
    x_tr, x_te = X_train.copy()[train_ind], X_train.copy()[test_ind]
    y_tr, y_te = y_train.copy()[train_ind], y_train.copy()[test_ind]
    
    f.init(x_tr, y_tr)
    f.train(no_base_classifiers)
    y_predict = f.predict(x_te)
    accuracy = np.mean(y_predict == y_te)
    cv_acc_arr.append(accuracy)
    
    ## comparing sklearn implementation of boost
    boost.fit(x_tr,y_tr)
    y_pred = boost.predict(x_te)
    accuracy_sk = np.mean(y_pred == y_te)
    cv_sk_acc_arr.append(accuracy_sk)

    i += 1

print(np.mean(cv_acc_arr))
print(np.mean(cv_sk_acc_arr))
