import pandas as pd
import numpy as np


def prediction_probability(prediction, ydata):


def test_divide_DNN(data, key):
    ydata = data.loc[:, key].as_matrix()
    xdata = data.iloc[:, 1:-3].as_matrix()
    return xdata, ydata


def shuffle_index(data):
    randomnumbers = np.random.randint(1, 6, size=len(data))
    # print(randomnumbers)
    return randomnumbers



def fivefold(data, key ):
    """
    :param: data is whole data for trainning and testing
    :return: seperate data for five dataframe or list
    """
    data_five = []
    # print(data)

    for i in range(1, 6):
        data_five.append(data[data.loc[:, key] == i])
        # print(data_five[i-1])
    return data_five


def n_fold(data, key, n):
    """
    :param: data is whole data for trainning and testing
    :return: seperate data for five dataframe or list
    """
    data_n = []
    # print(data)
    ## key column에 있는 값에 따라 데이터를 n-등분 하는 식으로 구현
    keys = []


    return data_n


def divide_xy_train(data_n , key ,to_matrix , x_start, x_end):
    """
   :return: X and Y data
    """
    y_data_n = []
    x_data_n = []
    for data in data_n :
        #return NumpyArray#
        y_data_n.append(data.loc[:,key].as_matrix())
        x_data_n.append(data.iloc[:,x_start:x_end].as_matrix())

    return x_data_n , y_data_n


def divide_xy_test(data , key , x_start, x_end ):
    ydata = data.loc[:, key].as_matrix()
    xdata = data.iloc[:, x_start:x_end].as_matrix()
    return xdata , ydata



def train_test(datas, key):
    """
    :param: data is whole data for trainning and testing
    :return: seperate data for five dataframe or list
    """
    test_data = datas[key]
    train_data = []
    keychecker = 0
    count = 0
    for data in datas :
        if keychecker == key:
            pass
        else :
            if count == 0 :
                train_data = data
            else :
                train_data = np.concatenate( (train_data, data ), axis=0)
            count +=1
        keychecker += 1

    return train_data, test_data

def test_validation(data):
    ## 501개
    ## validation [ 0: 250 , :]
    ## test [250: , :]
    val_data = data[:len(data)//2,:]
    test_data = data[len(data)//2:, :]

    return val_data, test_data


def one_hot_encoder(data):
    # print(data)
    max_val = max(data)
    print("Number of Class :", max_val+1)
    result = np.zeros((len(data), max_val+1))
    result[np.arange(len(data)), data] = 1

    return result

def get_index_fromdataframe(data):
    return data['patient']

def get_result_fromdataframe(data):
    return data['result']
