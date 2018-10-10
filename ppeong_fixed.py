import pandas as pd
import tensorflow as tf
import keras
import os, sys, getopt, errno
import lib.dataProcess as dp
import lib.deepLearning as dl
import copy
from pandas import DataFrame as df

def check_correct(predict, y):
    result = {}
    result['cancer-correct'] = 0
    result['cancer-wrong'] = 0
    result['normal-correct'] = 0
    result['normal-wrong'] = 0

    for i in range(len(predict)) :
        if predict[i] == y[i] :
            if y[i] == 0 :
                result['normal-correct'] += 1
            else :
                result['cancer-correct'] += 1
        else :
            if y[i] == 0 :
                result['normal-wrong'] += 1
            else :
                result['cancer-wrong'] += 1

    for result_k, result_v in result.items():
        print(result_k +" : "+ str(result_v))

def make_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def print_help():
    print("This Program is for ensemble train. you should insert model numbers list")
    print("example : python3 ensemble_train.py 5 4 4 3 1")


raw_data = pd.read_csv("toy_ts_TCGA_with_GEO_input_ensemble_foundation_308.csv")
patient = list(raw_data.loc[:, 'patient'])
genenames = list(raw_data)[1:-2]
cancercode = list(raw_data.loc[:, 'cancer_code'])

test_x, test_y = dp.divide_xy_test(raw_data, 'result', 1, -2)
test_y1 = dp.one_hot_encoder(test_y)

print("DATA PROCESSING END")

test_prob = pd.DataFrame()

l_model = keras.models.load_model("m_3.h5")

test_h=l_model.predict(test_x, batch_size=None, verbose=0, steps=None)


test_h

num_of_genes=int(test_x.shape[1])
num_of_samples=int(test_x.shape[0])

num_of_samples

for p in range(num_of_samples):
    print("sample: ",p,"model: m_3")
    cc = cancercode[p]
    result={}
    for g in range(num_of_genes):
        print("gene: m_3")
        ppeong_test_x = copy.deepcopy(test_x)

        min_gene_expression = min(ppeong_test_x[:, g])
        max_gene_expression = max(ppeong_test_x[:, g])
        original_gene_expression = ppeong_test_x[p, g]

        iteration = 10
        step = (max_gene_expression-min_gene_expression)/(iteration-1)

        prob_change = []
        min_prob = 1
        max_prob = 0
        for j in range(iteration):
            changed_gene_expression = min_gene_expression+step*j
            # change expression
            ppeong_test_x[:, g] = changed_gene_expression # error? confirm
            # predict cancer probability of 'p'th patient
            ppeong_test_h = l_model.predict(ppeong_test_x[p:p+1,:], batch_size=None, verbose=0, steps=None)
            #'''
            print("################################################### for debug ###################################################")
            print(ppeong_test_h)
            #'''

            new_prob = ppeong_test_h[0,0]### new prob of being cancer
            if min_prob >= new_prob:
                min_prob = new_prob
            if max_prob <= new_prob:
                max_prob = new_prob

            prob_change.append(new_prob)

        # gene impact = original cancer prob - min cancer prob
        gene_impact = test_h[p, 0]-min_prob
                                  
        prob_change.append(test_h[p, 0])
        prob_change.append(test_y[p])
        prob_change.append(original_gene_expression)
        prob_change.append(min_gene_expression)
        prob_change.append(max_gene_expression)
        prob_change.append(gene_impact)

        result[genenames[g]] = prob_change

        output_file = pd.DataFrame.from_dict(result,orient = 'index')
        col_names = ["step_"+str(i) for i in range(iteration)]
        col_names.append("original_prob")
        col_names.append("real_Y")
        col_names.append("original_x")
        col_names.append("min_x")
        col_names.append("max_x")
        col_names.append("gene_impact")

        output_file.columns = col_names

        filename = "ppeong_model_"+str(patient[p])+"_"+str(cc)+".csv"
        print(filename)
        output = "C:/Users/tkddl/Downloads/Example_DeepLearning/ppeong_"+filename
        print(output)
        output_file.to_csv("./ppeong/ppeong_model_"+str(patient[p])+"_"+str(cc)+".csv")
