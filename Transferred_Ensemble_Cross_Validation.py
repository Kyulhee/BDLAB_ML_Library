#!/usr/bin/env python
# coding: utf-8

# # Importing Library

# In[6]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Input, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import Model, load_model, Sequential 

early_stopping = EarlyStopping(patience=10)

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from pandas import DataFrame as df

np.random.seed(777)

print(tf.__version__)


# ##  Functions library

# In[7]:


# prediction
def check_correct(predict, y):
    result = {}
    result['resistant-correct'] = 0
    result['resistant-wrong'] = 0
    result['sensitive-correct'] = 0
    result['sensitive-wrong'] = 0

    for i in range(len(predict)) :
        if predict[i] == y[i] :
            if y[i] == 0 :
                result['sensitive-correct'] += 1
            else :
                result['resistant-correct'] += 1
        else :
            if y[i] == 0 :
                result['sensitive-wrong'] += 1
            else :
                result['resistant-wrong'] += 1

    #for result_k, result_v in result.items():
    #    print(result_k +" : "+ str(result_v))
    sensitivity=result['resistant-correct']/(result['resistant-correct']+result['resistant-wrong'])
    specificity=result['sensitive-correct']/(result['sensitive-correct']+result['sensitive-wrong'])
    #print("Sensitivity :", sensitivity)
    #print("Specificity :", specificity)
    return sensitivity, specificity


# In[8]:


# devide raw data into train / test & x_val / y_val
def data_split(raw_data, index_col, test_index):
    
    train_data = raw_data.iloc[list(raw_data.iloc[:,index_col]!=test_index)]
    test_data = raw_data.iloc[list(raw_data.iloc[:,index_col]==test_index)]
    
    y_val = train_data.Platinum_Status
    x_val = train_data.drop(["Platinum_Status","index"],axis=1)
    test_y_val = test_data.Platinum_Status
    test_x_val = test_data.drop(["Platinum_Status","index"],axis=1)
    
    return train_data, test_data, y_val, x_val, test_y_val, test_x_val

    # raw_data: have gene_expressions(maybe multiple columns), index column, Platinum_Status column.


# In[9]:


# calculate all of model performance 
# - predictions(probability) / labeled predictions(0/1) / Loss / Accuracy / Sensitivity / Specificity / AUC values of Train / Test dataset.
# using trained models, or you can put predictions(probability) passively(in this case, Loss & Accuracy do not provided.)
def model_performance(information=False, Input_Prediction_Passively=False, using_model=None, tr_predictions=None, ts_predictions=None, tr_x_val=None, tr_y_val=None, ts_x_val=None, ts_y_val=None, output_list=None):
    
    if information == True:            
        print("options model_performance:\n1) using_model: keras models that you want to check performance. \"Input_Prediction_Passive\" option for input prediction list instead using models.\n3) tr_predictions & ts_predictions: prediction input passively. put this data only when not using keras model.\n4) tr_x_val & ts_x_val: input samples of train/test samples.\n4) tr_y_val & ts_y_val: results of train/test samples.\n5) output_list: return values that you want to recieve.\n CAUTION: Essential variable.\n\t tr_loss, tr_accuracy, tr_sensitivity, tr_specificity, tr_predictions, labeled_tr_predictions, tr_predictions_flat, roc_auc_tr,\nts_loss, ts_accuracy, ts_sensitivity, ts_specificity, ts_predictions, labeled_ts_predictions, ts_predictions_flat, roc_auc_ts,\nroc_auc_total\n\n* CAUTION: if 'None' value is returned, please check your input tr inputs(None value for tr outputs) or ts inputs(None value for ts outputs).") 
        return 0
    elif information != False:
        print("for using information options, please set 'information' variable for 'True'")
        return -1
    
    if using_model is None:
        if Input_Prediction_Passively == False:
            print("ERROR: There are no models for using.\nusing \"model_performance(information = True)\" for getting informations of this function.") 
            return -1
        elif (tr_predictions is None) and (ts_predictions is None): # No model/prediction input. no performance should be calculated.
                print("ERROR: Input prediction list instead using saved model.")
                return -1
        else: # No model input, but Input_Prediction_Passively is True & input prediction is valid.
            tr_loss,tr_accuracy= None, None
            ts_loss,ts_accuracy= None, None
            
    elif Input_Prediction_Passively == True: # both of model/prediction putted, could cause confusing.
        ch = input("You put both model and prediction. Select one method:\n'p' for using prediction only, 'm' using models only, 'n' for quit the function.")
        while 1:
            if ch == 'p':
                using_model = None
                break
            elif ch == 'm':
                tr_predictions = None
                ts_predictions = None
                break
            elif ch == 'e':
                return 0
            else:
                print("you put worng option: "+str(ch))
            ch = input("Select one method:\n'p' for using prediction only, 'm' using models only, 'n' for quit the function.")
                
    if output_list is None:
        print("ERROR: There are no output_list for return.\nusing \"model_performance(information = True)\" for getting informations of this function.")
        return -1
    
    if not(tr_x_val is None) and not(tr_y_val is None):
        # predict tr result only when no tr_prediction input
        if tr_predictions is None:
            tr_loss,tr_accuracy= using_model.evaluate(tr_x_val,tr_y_val)
            tr_predictions = using_model.predict(tr_x_val)
        # tr sensitivity / specificity
        labeled_tr_predictions = np.where(tr_predictions > 0.5, 1, 0).flatten()
        tr_sensitivity, tr_specificity = check_correct(labeled_tr_predictions, tr_y_val)
        tr_predictions_flat = tr_predictions[:,0]   
        # roc(tr)
        fpr_tr, tpr_tr, threshold_tr = metrics.roc_curve(tr_y_val, tr_predictions)
        roc_auc_tr = metrics.auc(fpr_tr, tpr_tr)
    
    if not(ts_x_val is None) and not(ts_y_val is None):
        # predict ts result only when no ts_prediction input
        if ts_predictions is None:
            ts_loss,ts_accuracy= using_model.evaluate(ts_x_val,ts_y_val)
            ts_predictions = using_model.predict(ts_x_val)
        labeled_ts_predictions = np.where(ts_predictions > 0.5, 1, 0).flatten()
        ts_sensitivity, ts_specificity = check_correct(labeled_ts_predictions, ts_y_val)
        ts_predictions_flat = ts_predictions[:,0]   
        # roc(ts)
        fpr_ts, tpr_ts, threshold_ts = metrics.roc_curve(ts_y_val, ts_predictions)
        roc_auc_ts = metrics.auc(fpr_ts, tpr_ts)    
    
    if (not(tr_x_val is None) and not(tr_y_val is None)) and (not(ts_x_val is None) and not(ts_y_val is None)):
        y_true = np.append(tr_y_val, ts_y_val)
        y_pred = np.append(tr_predictions, ts_predictions)
        fpr_total, tpr_total, threshold_total = metrics.roc_curve(y_true, y_pred)
        roc_auc_total = metrics.auc(fpr_total, tpr_total)
        
        
    return_list = []
    
    for output in output_list:
        
        if(output == "tr_loss"):
            return_list.append(tr_loss)
                               
        elif(output == "tr_accuracy"):
            return_list.append(tr_accuracy)
                               
        elif(output == "tr_sensitivity"):
            return_list.append(tr_sensitivity)
                               
        elif(output == "tr_specificity"):
            return_list.append(tr_specificity)
                               
        elif(output == "tr_predictions"):
            return_list.append(tr_predictions)
                               
        elif(output == "labeled_tr_predictions"):
            return_list.append(labeled_tr_predictions)
                               
        elif(output == "tr_predictions_flat"):
            return_list.append(tr_predictions_flat)
            
        elif(output == "roc_auc_tr"):
            return_list.append(roc_auc_tr)

        elif(output == "ts_loss"):
            return_list.append(ts_loss)
                               
        elif(output == "ts_accuracy"):
            return_list.append(ts_accuracy)
                               
        elif(output == "ts_sensitivity"):
            return_list.append(ts_sensitivity)
                               
        elif(output == "ts_specificity"):
            return_list.append(ts_specificity)
                               
        elif(output == "ts_predictions"):
            return_list.append(ts_predictions)
                               
        elif(output == "labeled_ts_predictions"):
            return_list.append(labeled_ts_predictions)
                               
        elif(output == "ts_predictions_flat"):
            return_list.append(ts_predictions_flat)
        
        elif(output == "roc_auc_ts"):
            return_list.append(roc_auc_ts)
            
        elif(output == "roc_auc_total"):
            return_list.append(roc_auc_total)
                               
        else:
            print("There are no options <"+str(output)+">. Please refer these output options:\ntr_loss, tr_accuracy, tr_sensitivity, tr_specificity, tr_predictions, labeled_tr_predictions, tr_predictions_flat, roc_auc_tr,\nts_loss, ts_accuracy, ts_sensitivity, ts_specificity, ts_predictions, labeled_ts_predictions, ts_predictions_flat, roc_auc_ts,\nroc_auc_total")
    
    return return_list


# # 1. Preparation: import & preprocessing data + import module

# ## Input path & name of models / raw data for ensemble

# In[10]:


types = ["OV_six_fold_Annotation3000_100", 
         "OV_six_fold_CV_100", 
         "OV_six_fold_Var_100", "OV_six_fold_new_Diff_100",
         "OV_six_fold_Clin", 
         "OV_six_fold_SNV" 
         ]

# input pathes
path = "C:/test/TC_six_fold_subsamples/"
save_model_path = "../models/Ovary"
save_prediction_path = "../result/Ovary"


# ## Import Data

# In[11]:


file_1 = path+types[0]+".csv"
file_2 = path+types[1]+".csv"
file_3 = path+types[2]+".csv"
file_4 = path+types[3]+".csv"
file_5 = path+types[4]+".csv"
file_6 = path+types[5]+".csv"

idx_col = 0

data_1 = pd.read_csv(file_1,index_col=idx_col)
data_2 = pd.read_csv(file_2,index_col=idx_col)
data_3 = pd.read_csv(file_3,index_col=idx_col)
data_4 = pd.read_csv(file_4,index_col=idx_col)
data_5 = pd.read_csv(file_5,index_col=idx_col)
data_6 = pd.read_csv(file_6,index_col=idx_col)

inter_data_1 = data_1.iloc[list(data_1.iloc[:,-1]!=6)]
inter_data_2 = data_2.iloc[list(data_2.iloc[:,-1]!=6)]
inter_data_3 = data_3.iloc[list(data_3.iloc[:,-1]!=6)]
inter_data_4 = data_4.iloc[list(data_4.iloc[:,-1]!=6)]
inter_data_5 = data_5.iloc[list(data_5.iloc[:,-1]!=6)]
inter_data_6 = data_6.iloc[list(data_6.iloc[:,-1]!=6)]

data_list  = [data_1,  data_2,  data_3,  data_4, data_5,  data_6]
inter_data_list  = [inter_data_1,  inter_data_2,  inter_data_3,  inter_data_4, inter_data_5,  inter_data_6]

# for selection
select = [3, 4, 4]

select_types = []
select_data = []
select_inter_data = []
parameter_list = {"lr":[0.003, 0.001, 0.001], "batch_size" :[5, 10, 5],"Batch_Normalize":[False, False, True], "input_drop_out":[0.5,0.5,0.3], "drop_out":[0.5,0.5,0],  "layers":[[150,200,200], [100, 200, 200], [100]]}

model_num_list = []
test_index_list = []
tr_acc_list = []
ts_acc_list = []
tr_sensitivity_list = []
ts_sensitivity_list = []
tr_specificity_list = []
ts_specificity_list = []
tr_auc_list = []
ts_auc_list = []
tot_auc_list = []

for s in select:
    select_types.append(types[s-1])
    select_data.append(data_list[s-1])
    select_inter_data.append(inter_data_list[s-1])
    print(str(len(select_types))+"file_name: "+ types[s-1]+"\nsample(full) : "+str(data_list[s-1].shape[0])+"\nsample(inter) : "+str(inter_data_list[s-1].shape[0])+"\nfeatures : "+str(data_list[s-1].shape[1]))
    

for ts_i in range(1,6):
    
    print("test index: " + str(ts_i))
    dataset = {"train_data":[], "test_data":[], "tr_y_val":[], "tr_x_val":[], "ts_y_val":[], "ts_x_val":[]}
    val_name = ["train_data", "test_data", "tr_y_val", "tr_x_val", "ts_y_val", "ts_x_val"]

    for s_d in select_data:
        train_data, test_data, tr_y_val, tr_x_val, ts_y_val, ts_x_val = data_split(raw_data = s_d, index_col = -1, test_index = ts_i)
        dataset['train_data'].append(train_data)
        dataset['test_data'].append(test_data)
        dataset['tr_x_val'].append(tr_x_val)
        dataset['tr_y_val'].append(tr_y_val)
        dataset['ts_x_val'].append(ts_x_val)
        dataset['ts_y_val'].append(ts_y_val)        
        
    model_list = []
    for  t in range(len(select)):
        print("Model "+str(t+1)+": "+str(select_types[t]))

    #select data

    for i in range(len(select)):
       
        features = dataset['tr_x_val'][i].shape[1]
        # 1) parameter setting
        adam = optimizers.Adam(lr=parameter_list["lr"][i])                                   
        input_drop_out_m = parameter_list["input_drop_out"][i]
        drop_out_m = parameter_list["drop_out"][i]
        BN = parameter_list["Batch_Normalize"][i]                           
        
        layers = parameter_list["layers"][i]
        m_tr_loss_best = 100 # for saving best loss value 
        best_m_model=[] #for saving best model
        count=0 # for early stopping

        # 2) model build
        input_m = Input(shape=(features,))
        m_m_dp = Dropout(input_drop_out_m)(input_m)
        for l in layers:
            if BN == True:
                m_m = Dense(l)(m_m_dp)
                m_m_bn = BatchNormalization(axis=1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')(m_m)
                m_m_dp = Activation("relu")(m_m_bn)
            else:
                m_m = Dense(l,activation='relu')(m_m_dp)
                m_m_dp = Dropout(drop_out_m)(m_m)
                               
        m_m_final = m_m_dp
        output_m = Dense(1, activation="sigmoid")(m_m_final)
        m_model = Model(inputs=input_m,outputs=output_m)
        m_model.compile(optimizer=adam, 
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        # 3) Training: if no increase of tr_loss three times, stop training.
        while 1:
            m_model.fit(dataset['tr_x_val'][i], dataset['tr_y_val'][i], batch_size=parameter_list["batch_size"][i], nb_epoch=1)
            m_tr_loss=m_model.evaluate( dataset['tr_x_val'][i], dataset['tr_y_val'][i])[0]
            if m_tr_loss < m_tr_loss_best: # new best model. count reset.
                m_tr_loss_best = m_tr_loss
                count=0
                best_m_model = m_model
            if count>20: # no increase 20 time. stop.
                m_model = best_m_model
                break
            else: count=count+1
        print("Model " + str(i)+"-"+str(ts_i) + " trained.")

        # 4) save model
        model_list.append(m_model)
        m_model.save(save_model_path+"/m_"+str(i)+"-"+str(ts_i)+".h5")
        print("Model "+ str(i)+"-"+str(ts_i) + " saved.")

        # 5) evaluate model
        m_output_list = model_performance(
            information = False, using_model=m_model,Input_Prediction_Passively = False, 
            tr_x_val=dataset['tr_x_val'][i], tr_y_val=dataset['tr_y_val'][i], ts_x_val=dataset['ts_x_val'][i], ts_y_val=dataset['ts_y_val'][i],
            output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                         "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                         "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                         "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                         "roc_auc_total"])

        m_tr_loss, m_tr_accuracy, m_tr_sensitivity, m_tr_specificity, m_tr_predictions, m_labeled_tr_predictions, m_tr_predictions_flat, m_roc_auc_tr, m_ts_loss, m_ts_accuracy, m_ts_sensitivity, m_ts_specificity, m_ts_predictions,m_labeled_ts_predictions, m_ts_predictions_flat, m_roc_auc_ts, m_roc_auc_total = m_output_list

        print("Overall AUC: ", m_roc_auc_total)
        print("Train AUC: ", m_roc_auc_tr)
        print("Test AUC: ", m_roc_auc_ts)

        print("Train Accuracy: {}".format(m_tr_accuracy))
        print("Train Sensitivities & Specificities : "+str(m_tr_sensitivity)+", "+str(m_tr_specificity))
        print("Test Accuracy: {}".format(m_ts_accuracy))
        print("Test Sensitivities & Specificities : "+str(m_ts_sensitivity)+", "+str(m_ts_specificity))

        model_num_list.append(i)
        test_index_list.append(ts_i)
        tr_acc_list.append(m_tr_accuracy)
        ts_acc_list.append(m_ts_accuracy)
        tr_sensitivity_list.append(m_tr_sensitivity)
        ts_sensitivity_list.append(m_ts_sensitivity)
        tr_specificity_list.append(m_tr_specificity)
        ts_specificity_list.append(m_ts_specificity)
        tr_auc_list.append(m_roc_auc_tr)
        ts_auc_list.append(m_roc_auc_ts)
        tot_auc_list.append(m_roc_auc_total)

        # save prediction result.

        tr_df_m = pd.DataFrame(data={"patient":list(dataset['train_data'][i].index), "hypothesis 1": list(m_tr_predictions_flat), 
                                "prediction":list(m_labeled_tr_predictions), "Platinum_Status":list(dataset['tr_y_val'][i])}) 
        tr_df_m.to_csv(save_prediction_path+"prediction_result_m_"+str(i)+"-"+str(ts_i)+"_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

        ts_df_m = pd.DataFrame(data={"patient":list(dataset['test_data'][i].index), "hypothesis 1": list(m_ts_predictions_flat), 
                                "prediction":list(m_labeled_ts_predictions), "Platinum_Status":list(dataset['ts_y_val'][i])})
        ts_df_m.to_csv(save_prediction_path+"prediction_result_m_"+str(i)+"-"+str(ts_i)+"_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

df = df(data = {'test_index':test_index_list , 'model_numbers':model_num_list, 'tr_accuracy':tr_acc_list, 'tr_sensitivity':tr_sensitivity_list, 'tr_specificity':tr_specificity_list, 'ts_accuracy': ts_acc_list, 'ts_sensitivity':ts_sensitivity_list, 'ts_specificity':ts_specificity_list, "tr_auc":tr_auc_list, "ts_auc":ts_auc_list, "total_auc":tot_auc_list}, columns =['test_index', 'model_numbers', 'tr_accuracy', 'tr_sensitivity', 'tr_specificity', 'ts_accuracy', 'ts_sensitivity', 'ts_specificity', "tr_auc", "ts_auc", "total_auc"])
