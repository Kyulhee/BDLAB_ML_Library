import pandas as pd
import tensorflow as tf
import keras
import os, sys, getopt, errno
import lib.dataProcess as dp
import lib.deepLearning as dl
import copy
from pandas import DataFrame as df

def test_divide_DNN(data, key, start, end):
    ydata = data.loc[:, key].as_matrix()
    xdata = data.iloc[:, start:end].as_matrix()
    return xdata, ydata

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


model_numbers=[]
#if __name__ == "__main__":
#model_numbers = sys.argv[1:]
while(1):
    ch = input("model number('q' for stop): ")
    if (ch =='q'):
        break
    else:
        model_numbers.append(int(ch))
if len(model_numbers) == 0:
    raise ValueError("Model Numbers have to be >1.")
for i in range(len(model_numbers)):
    model_numbers[i] = int(model_numbers[i])

input_directory = 'C:/test/ppeong/'
model_directory = 'C:/test/ppeong/'
output_dir = "C:/test/ppeong/output/"
'''
input_directory = '/home/tjahn/Data/input_ensemble/'
model_directory = './model/'
output_dir = 'home/tjahn/Git2/Users/kyulhee/'
'''
'''data processing'''
models = []
inputfiles = []
for i in range(len(model_numbers)):
    models.append(model_directory + 'model_' + str(model_numbers[i]) + '.h5')
    inputfiles.append(input_directory + 'selected_model_' + str(model_numbers[i]) + '_data.csv')

test_x_list = []
test_y_list = []
genename_list = []
patient_list = []
cancercode_list = []

for inputfile in inputfiles:
    raw_data = pd.read_csv(inputfile)
    patient = list(raw_data.loc[:, 'patient'])
    genenames = list(raw_data)[1:-2]
    cancercode = list(raw_data.loc[:, 'cancer_code'])
    test_x, test_y = dp.test_divide_DNN(raw_data, 'result', 1, -2)
    test_y = dp.one_hot_encoder(test_y)

    test_x_list.append(test_x)
    test_y_list.append(test_y)
    genename_list.append(genenames)
    patient_list.append(patient)
    cancercode_list.append(cancercode)

print("DATA PROCESSING END")

test_prob = pd.DataFrame()
        ### Peong Start ####
for model_num in range(len(model_numbers)):

    test_x = test_x_list[model_num]
    test_y = test_y_list[model_num]
    genenames = genename_list[model_num]
    patient = patient_list[model_num]
    cancercode = cancercode_list[model_num]

    l_model = keras.models.load_model(models[model_num])
    # original probability table of whole patients
    test_h = l_model.predict(test_x, batch_size=None, verbose=0, steps=None)
    
    num_of_genes=int(test_x.shape[1])
    num_of_samples=int(test_x.shape[0])

    for p in range(num_of_samples):
        print("sample: "+patient[p]+ ", " + str(p+1)+"/"+str(num_of_samples))
        cc = cancercode[p]
        result={}
        for g in range(num_of_genes):
            #print("gene: ",genenames[g], ", ", str(g+1), "/", str(num_of_genes))
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
                ppeong_test_x[:, g] = changed_gene_expression
                # predict cancer probability of 'p'th patient
                ppeong_test_h = l_model.predict(ppeong_test_x[p:p+1,:], batch_size=None, verbose=0, steps=None)
                '''
                print("################################################### for debug ###################################################")
                print(ppeong_test_h)
                '''
    
                new_prob = ppeong_test_h[0,0]### new prob of being cancer
                if min_prob >= new_prob:
                    min_prob = new_prob
                    min_prob_expression = changed_gene_expression
                if max_prob <= new_prob:
                    max_prob = new_prob

                prob_change.append(new_prob)

            # gene impact = original cancer prob - min cancer prob
            gene_impact = test_h[p, 0]-min_prob
            # gene impact vector
            gene_impact_vector = gene_impact * (abs(original_gene_expression - min_prob_expression) / ( original_gene_expression - min_prob_expression))
                                      
            prob_change.append(test_h[p, 0])
            prob_change.append(test_y[p,0])
            prob_change.append(original_gene_expression)
            prob_change.append(min_gene_expression)
            prob_change.append(max_gene_expression)
            prob_change.append(gene_impact)
            prob_change.append(gene_impact_vector)

            result[genenames[g]] = prob_change

        output_file = pd.DataFrame.from_dict(result,orient = 'index')
        col_names = ["step_"+str(i) for i in range(iteration)]
        col_names.append("original_prob")
        col_names.append("real_Y")
        col_names.append("original_x")
        col_names.append("min_x")
        col_names.append("max_x")
        col_names.append("gene_impact")
        col_names.append("gene_impact_vector")

        output_file.columns = col_names

        filename = "ppeong_model"+str(model_num)+"_"+str(patient[p])+"_"+str(cc)+".csv"
        print(filename)
        output = output_dir+filename
        #print(output)
        output_file.to_csv(output,sep=",")
