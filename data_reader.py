## This file contains functions that read the abalone dataset and format it 
## into the right form to input to an svm

import numpy as np
import pandas as pd

# Reading data

class AbloneDataReader:

    def __init__(self, no_train_samples, no_test_samples):
        column_names = ["sex", "length", "diameter", "height", "whole weight", 
                        "shucked weight", "viscera weight", "shell weight", "rings"]

        sex_categories = {"M" : 0, "F" : 1, "I" : 2}

        data = pd.read_csv("./data/abalone.data", names = column_names)

        # transforming data to categorical form

        # 3 because the "sex" category will be converted to categorical form 
        # 7 for the remaining attriutes
        # 1 for the binary labels
        data_arr = np.zeros((len(data["sex"]), 3 + 7 + 1))

        for i in range(0,8):
            data_arr[:,i+3] = data[column_names[i+1]]
        
        for i in range(len(data["sex"])):
            ## converting to categorical form M = (1,0,0), F = (0, 1 0), I = (0, 0, 1)
            data_arr[i][sex_categories[data["sex"][i]]] = 1.0
            # transforming the 29 classes to a binary class problem
            if data["rings"][i] > 9:
                data_arr[i][10] = 1
            else:
                data_arr[i][10] = -1
        
        self.data_train = data_arr[0:no_train_samples]
        self.data_test = data_arr[no_train_samples:no_train_samples + no_test_samples]

    def get_train_data(self):

        x_train = self.data_train.copy()[:,0:10]
        y_train = self.data_train.copy()[:,10]

        return x_train, y_train

    def get_test_data(self):

        x_test = self.data_test.copy()[:,0:10]
        y_test = self.data_test.copy()[:,10]

        return x_test, y_test
        
    