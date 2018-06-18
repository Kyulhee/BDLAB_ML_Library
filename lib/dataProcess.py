import pandas as pd
import numpy as np
import time

def prediction_probability(hypothesis, prediction, ydata, ids):
    try :
        if len(prediction) != len(ydata) or len(ydata) != len(ids) or len(ydata) != len(hypothesis):
            raise ValueError
        result = ids.to_frame('ID')
        result = pd.concat([result, pd.DataFrame(hypothesis, columns=['hypothesis'+str(i) for i in len(hypothesis)]), pd.DataFrame(prediction, columns='prediction') , pd.DataFrame(ydata , columns='label') ] ,axis=1 )
        output_f_name = time.strftime("%d/%m/%Y")+time.strftime("%H:%M:%S")+'result.csv'
        result.to_csv(output_f_name , index=False)
    except ValueError :
        print("Cannot make prediction-probability file because length of arguments are not same.")


def test_divide_DNN(data, key, start, end):
    ydata = data.loc[:, key].as_matrix()
    xdata = data.iloc[:, start:end].as_matrix()
    return xdata, ydata


def shuffle_index(data):
    randomnumbers = np.random.randint(1, 6, size=len(data))
    # print(randomnumbers)
    return randomnumbers

def fivefold(data, key ):
    """
    :function:
    :param:
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
    max_val = max(data.loc[:, key])
    try :
        if max_val != n :
            raise ValueError
        for i in range(1, n):
            selected_data = data[data.loc[:, key] == i]
            if selected_data.empty :
                raise TypeError
            data_n.append(selected_data)

    except ValueError :
        print("Value Error : Max value of key column is not same with argument n")
    except TypeError:
        print("DataFrame selected by key is Empty")

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
