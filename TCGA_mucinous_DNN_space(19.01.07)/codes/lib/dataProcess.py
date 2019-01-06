import pandas as pd
import numpy as np
import time
from pandas import DataFrame as df

def mk_t_data(raw_data, column_name):
    t_data=raw_data.transpose()
    t_data.columns=t_data.iloc[0]
    t_data=t_data.iloc[1:,:]
    df_index=df(t_data.index)
    df_index.index=t_data.index
    df_index.columns=[column_name]
    t_data=pd.concat([df_index,t_data],axis=1)
  
    return t_data

def mk_cut_data(raw_data, pathway):
    
    t_raw_data=raw_data.transpose()
    t_raw_data.columns=t_raw_data.iloc[0]
    t_raw_data=t_raw_data.iloc[1:,:]
    df_gene_name=df(t_raw_data.index)
    t_raw_data=t_raw_data.reset_index(drop=True)
    df_gene_name.rename(columns={0:"sample"}, inplace = True)
    df_gene_name=df_gene_name.reset_index(drop=True)
    t_raw_data=pd.concat([df_gene_name,t_raw_data],axis=1)
    
    genes=[]
    for key in pathway.keys():
        for i in range(len(pathway[key])):
            genes.append(pathway[key][i])

    df_genes=df(genes)

    dup_genes=df(genes).drop_duplicates()
    dup_genes=dup_genes.reset_index(drop=True)

    cut_gene=[]
    for i in range(len(dup_genes)):
        cut_gene.append(dup_genes[0][i])
    cut_gene.append('result')

    df_cut_gene=df(cut_gene)

    t_cut_data=df(t_raw_data[t_raw_data['sample'].isin(df_cut_gene[0].as_matrix())])

    t_cut_data=t_cut_data.transpose()
    t_cut_data.columns=t_cut_data.iloc[0]
    t_cut_data=t_cut_data.iloc[1:,:]
    df_cut_index=df(t_cut_data.index)
    df_cut_index.rename(columns={0:"sample"}, inplace = True)
    df_cut_index.index=t_cut_data.index
    data=pd.concat([df_cut_index,t_cut_data],axis=1)
    data=data.reset_index(drop=True)
    
    return data

def gmt_to_dictionary(filename):
    geneset = open(filename, 'r')
    lines = geneset.readlines()
    geneset.close()
    gene_names = []
    genes = []
    for line in lines:
        geneset = line.split('\t')
        gene_names.append(geneset[0])
        if geneset[-1][-1] == '\n':
            geneset[-1] = geneset[-1][:-1]
        genes.append(geneset[2:])

    gene_sets = dict(zip(gene_names, genes))
    return gene_sets

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

def twofold(data, key ):
    """
    :param: data is whole data for trainning and testing
    :return: seperate data for five dataframe or list
    """
    data_two = []
    # print(data)
    for i in range(1, 3):
        data_two.append(data[data.loc[:, key] == i])
        # print(data_five[i-1])
    return data_two



def divide_xy_train(data_five , key ,to_matrix , x_start, x_end):
    """
   :return: X and Y data
    """
    data = data_five
    y_data_five = []
    x_data_five = []
    for data in data_five :
        if to_matrix :
            #return NumpyArray#
            y_data_five.append(data.loc[:,key].as_matrix())
            x_data_five.append(data.iloc[:,x_start:x_end].as_matrix())
        else :
            ##return DataFrame#
            y_data_five.append(data.loc[:, key].as_matrix())
            x_data_five.append(data.iloc[:, x_start:x_end])
    return x_data_five , y_data_five


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
    val_data = data[:len(data)//2,:]
    test_data = data[len(data)//2:, :]

    return val_data, test_data


def one_hot_encoder(data):
    # print(data)
    max_val = max(data)
    result = np.zeros((len(data), max_val+1))
    data=np.int64(data)
    result[np.arange(len(data)), data] = 1

    return result

def one_hot_encoder_train(data):
    # print(data)
    max_val = max(data)
    result = np.zeros((len(data), int(max_val)+1))
    data=np.int64(data)
    result[np.arange(len(data)), data] = 1

    return result

def one_hot_encoder_test(data1,data2):
    max_val=max(data1)
    result=np.zeros((len(data2),int(max_val)+1))
    data2=np.int64(data2)
    result[np.arange(len(data2)),data2]=1
    
    return result

def get_index_fromdataframe(data):
    return data['patient']

def get_result_fromdataframe(data):
    return data['result']
