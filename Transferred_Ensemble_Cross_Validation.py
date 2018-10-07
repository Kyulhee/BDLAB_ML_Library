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
         "OV_Clin_ch_six_fold", 
         "OV_SNV_six_fold_sam" 
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
select = [1, 3, 4, 5]

select_types = []
select_data = []
select_inter_data = []

for i in select:
    select_types.append(types[i-1])
    select_data.append(data_list[i-1])
    select_inter_data.append(inter_data_list[i-1])
    print(str(len(select_types))+"file_name: "+ types[i-1]+"\nsample(full) : "+str(data_list[i-1].shape[0])+"\nsample(inter) : "+str(inter_data_list[i-1].shape[0])+"\nfeatures : "+str(data_list[i-1].shape[1]))
        

for ts_i in range(1,6):
    
    dataset = {"train_data":[], "test_data":[], "tr_y_val":[], "tr_x_val":[], "ts_y_val":[], "ts_x_val":[]}
    val_name = ["train_data":, "test_data", "tr_y_val", "tr_x_val", "ts_y_val", "ts_x_val"]

    parameter = {"lr"=[], "Batch_Normalize"=[], "input_dropout"=[], "drop_out"=[],  "layers "= []}
    
    for s_d in select_data:
        train_data, test_data, tr_y_val, tr_x_val, ts_y_val, ts_x_val = data_split(raw_data = s_d, index_col = -1, test_index = ts_i)
        dataset['train_data'].append(train_data)
        dataset['test_data'].append(test_data)
        dataset['tr_x_val'].append(tr_x_val)
        dataset['tr_y_val'].append(tr_y_val)
        dataset['ts_x_val'].append(ts_x_va)
        dataset['ts_y_val'].append(ts_y_val')

    for i in select:
        

    model_list = []
    for  t in range(len(select)):
        print("Model "+str(select+1)+": "+str(select_types[t]))

    #select data


        

    # 1) parameter setting
    adam = optimizers.Adam(lr=0.01)
    input_drop_out_m_1 = 0.3
    drop_out_m_1 = 0.5
    layers = [5]
    m_1_tr_loss_best = 100 # for saving best loss value 
    best_m_1_model=[] #for saving best model
    count=0 # for early stopping

    # 2) model build
    input_m_1 = Input(shape=(features_1,))
    m_1_m_dp = Dropout(input_drop_out_m_1)(input_m_1)
    for i in layers:
        m_1_m = Dense(i,activation='relu')(m_1_m_dp)
        m_1_m_dp = Dropout(drop_out_m_1)(m_1_m)
    m_1_m_final = m_1_m_dp
    output_m_1 = Dense(1, activation="sigmoid")(m_1_m_final)
    m_1_model = Model(inputs=input_m_1,outputs=output_m_1)
    m_1_model.compile(optimizer=adam, 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    # 3) Training: if no increase of tr_loss three times, stop training.
    while 1:
        m_1_model.fit(x_val_1, y_val_1, batch_size=5, nb_epoch=1)
        m_1_tr_loss=m_1_model.evaluate(x_val_1,y_val_1)[0]
        if m_1_tr_loss < m_1_tr_loss_best: # new best model. count reset.
            m_1_tr_loss_best = m_1_tr_loss
            count=0
            best_m_1_model = m_1_model
        if count>10: # no increase three time. stop.
            m_1_model = best_m_1_model
            break
        else: count=count+1
    print("Model 1 trained.")

    # 4) save model
    m_1_model.save(save_model_path+"/m_1.h5")
    print("Model 1 saved.")

    # 5) evaluate model
    m_1_output_list = model_performance(
        information = False, using_model=m_1_model,Input_Prediction_Passively = False, 
        tr_x_val=x_val_1, tr_y_val=y_val_1, ts_x_val=test_x_val_1, ts_y_val=test_y_val_1,
        output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                     "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                     "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                     "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                     "roc_auc_total"])

    m_1_tr_loss, m_1_tr_accuracy, m_1_tr_sensitivity, m_1_tr_specificity, m_1_tr_predictions, m_1_labeled_tr_predictions, m_1_tr_predictions_flat, m_1_roc_auc_tr, m_1_ts_loss, m_1_ts_accuracy, m_1_ts_sensitivity, m_1_ts_specificity, m_1_ts_predictions,m_1_labeled_ts_predictions, m_1_ts_predictions_flat, m_1_roc_auc_ts, m_1_roc_auc_total = m_1_output_list

    print("Overall AUC: ", m_1_roc_auc_total)
    print("Train AUC: ", m_1_roc_auc_tr)
    print("Test AUC: ", m_1_roc_auc_ts)

    print("Train Accuracy: {}".format(m_1_tr_accuracy))
    print("Train Sensitivities & Specificities : "+str(m_1_tr_sensitivity)+", "+str(m_1_tr_specificity))
    print("Test Accuracy: {}".format(m_1_ts_accuracy))
    print("Test Sensitivities & Specificities : "+str(m_1_ts_sensitivity)+", "+str(m_1_ts_specificity))




    # # Build & Evaluate models

    # ## 1) Model 1

    # In[12]:





    # In[ ]:


    # save prediction result.

    tr_df_m_1 = pd.DataFrame(data={"patient":list(train_data_1.index), "hypothesis 1": list(m_1_tr_predictions_flat), 
                            "prediction":list(m_1_labeled_tr_predictions), "Platinum_Status":list(y_val_1)})
    tr_df_m_1.to_csv(save_prediction_path+"prediction_result_m_1_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

    ts_df_m_1 = pd.DataFrame(data={"patient":list(test_data_1.index), "hypothesis 1": list(m_1_ts_predictions_flat), 
                            "prediction":list(m_1_labeled_ts_predictions), "Platinum_Status":list(test_y_val_1)})
    ts_df_m_1.to_csv(save_prediction_path+"prediction_result_m_1_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])


    # ## 2) Model 2

    # In[ ]:


    print("Model_2: "+str(types[1]))

    # 1) parameter setting
    adam = optimizers.Adam(lr=0.01)
    input_drop_out_m_2 = 0.3
    drop_out_m_2 = 0.5
    layers = [5]
    m_2_tr_loss_best = 100 # for saving best loss value 
    best_m_2_model=[] #for saving best model
    count=0 # for early stopping

    # 2) model build
    input_m_2 = Input(shape=(features_2,))
    m_2_m_dp = Dropout(input_drop_out_m_2)(input_m_2)
    for i in layers:
        m_2_m = Dense(i,activation='relu')(m_2_m_dp)
        m_2_m_dp = Dropout(drop_out_m_2)(m_2_m)
    m_2_m_final = m_2_m_dp
    output_m_2 = Dense(1, activation="sigmoid")(m_2_m_final)
    m_2_model = Model(inputs=input_m_2,outputs=output_m_2)
    m_2_model.compile(optimizer=adam, 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


    # 3) Training: if no increase of tr_loss three times, stop training.
    while 1:
        m_2_model.fit(x_val_2, y_val_2, batch_size=5, nb_epoch=1)
        m_2_tr_loss=m_2_model.evaluate(x_val_2,y_val_2)[0]
        if m_2_tr_loss < m_2_tr_loss_best: # new best model. count reset.
            m_2_tr_loss_best = m_2_tr_loss
            count=0
            best_m_2_model = m_2_model
        if count>3: # no increase three time. stop.
            m_2_model = best_m_2_model
            break
        else: count=count+1
    print("Model_2 trained.")

    # 4) save model
    m_2_model.save(save_model_path+"/m_2.h5")
    print("Model_2 saved.")

    # 5) evaluate model
    m_2_output_list = model_performance(
        information = False, using_model=m_2_model,Input_Prediction_Passively = False, 
        tr_x_val=x_val_2, tr_y_val=y_val_2, ts_x_val=test_x_val_2, ts_y_val=test_y_val_2,
        output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                     "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                     "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                     "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                     "roc_auc_total"])

    m_2_tr_loss, m_2_tr_accuracy, m_2_tr_sensitivity, m_2_tr_specificity, m_2_tr_predictions, m_2_labeled_tr_predictions, m_2_tr_predictions_flat, m_2_roc_auc_tr, m_2_ts_loss, m_2_ts_accuracy, m_2_ts_sensitivity, m_2_ts_specificity, m_2_ts_predictions,m_2_labeled_ts_predictions, m_2_ts_predictions_flat, m_2_roc_auc_ts, m_2_roc_auc_total = m_2_output_list

    print("Overall AUC: ", m_2_roc_auc_total)
    print("Train AUC: ", m_2_roc_auc_tr)
    print("Test AUC: ", m_2_roc_auc_ts)

    print("Train Accuracy: {}".format(m_2_tr_accuracy))
    print("Train Sensitivities & Specificities : "+str(m_2_tr_sensitivity)+", "+str(m_2_tr_specificity))
    print("Test Accuracy: {}".format(m_2_ts_accuracy))
    print("Test Sensitivities & Specificities : "+str(m_2_ts_sensitivity)+", "+str(m_2_ts_specificity))


    # In[ ]:


    # save prediction result.

    tr_df_m_2 = pd.DataFrame(data={"patient":list(train_data_2.index), "hypothesis 1": list(m_2_tr_predictions_flat), 
                            "prediction":list(m_2_labeled_tr_predictions), "Platinum_Status":list(y_val_2)})
    tr_df_m_2.to_csv(save_prediction_path+"prediction_result_m_2_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

    ts_df_m_2 = pd.DataFrame(data={"patient":list(test_data_2.index), "hypothesis 1": list(m_2_ts_predictions_flat), 
                            "prediction":list(m_2_labeled_ts_predictions), "Platinum_Status":list(test_y_val_2)})
    ts_df_m_2.to_csv(save_prediction_path+"prediction_result_m_2_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])


    # ## 3) Model 3 

    # In[ ]:


    print("Model_3: "+str(types[2]))

    # 1) parameter setting
    adam = optimizers.Adam(lr=0.01)
    input_drop_out_m_3 = 0.3
    drop_out_m_3 = 0.5
    layers = [5]
    m_3_tr_loss_best = 100 # for saving best loss value 
    best_m_3_model=[] #for saving best model
    count=0 # for early stopping

    # 2) model build
    input_m_3 = Input(shape=(features_3,))
    m_3_m_dp = Dropout(input_drop_out_m_3)(input_m_3)
    for i in layers:
        m_3_m = Dense(i,activation='relu')(m_3_m_dp)
        m_3_m_dp = Dropout(drop_out_m_3)(m_3_m)
    m_3_m_final = m_3_m_dp
    output_m_3 = Dense(1, activation="sigmoid")(m_3_m_final)
    m_3_model = Model(inputs=input_m_3,outputs=output_m_3)
    m_3_model.compile(optimizer=adam, 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


    # 3) Training: if no increase of tr_loss three times, stop training.
    while 1:
        m_3_model.fit(x_val_3, y_val_3, batch_size=5, nb_epoch=1)
        m_3_tr_loss=m_3_model.evaluate(x_val_3,y_val_3)[0]
        if m_3_tr_loss < m_3_tr_loss_best: # new best model. count reset.
            m_3_tr_loss_best = m_3_tr_loss
            count=0
            best_m_3_model = m_3_model
        if count>3: # no increase three time. stop.
            m_3_model = best_m_3_model
            break
        else: count=count+1
    print("Model_3 trained.")

    # 4) save model
    m_3_model.save(save_model_path+"/m_3.h5")
    print("Model_3 saved.")

    # 5) evaluate model
    m_3_output_list = model_performance(
        information = False, using_model=m_3_model,Input_Prediction_Passively = False, 
        tr_x_val=x_val_3, tr_y_val=y_val_3, ts_x_val=test_x_val_3, ts_y_val=test_y_val_3,
        output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                     "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                     "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                     "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                     "roc_auc_total"])

    m_3_tr_loss, m_3_tr_accuracy, m_3_tr_sensitivity, m_3_tr_specificity, m_3_tr_predictions, m_3_labeled_tr_predictions, m_3_tr_predictions_flat, m_3_roc_auc_tr, m_3_ts_loss, m_3_ts_accuracy, m_3_ts_sensitivity, m_3_ts_specificity, m_3_ts_predictions,m_3_labeled_ts_predictions, m_3_ts_predictions_flat, m_3_roc_auc_ts, m_3_roc_auc_total = m_3_output_list

    print("Overall AUC: ", m_3_roc_auc_total)
    print("Train AUC: ", m_3_roc_auc_tr)
    print("Test AUC: ", m_3_roc_auc_ts)

    print("Train Accuracy: {}".format(m_3_tr_accuracy))
    print("Train Sensitivities & Specificities : "+str(m_3_tr_sensitivity)+", "+str(m_3_tr_specificity))
    print("Test Accuracy: {}".format(m_3_ts_accuracy))
    print("Test Sensitivities & Specificities : "+str(m_3_ts_sensitivity)+", "+str(m_3_ts_specificity))


    # In[ ]:


    # save prediction result.

    tr_df_m_3 = pd.DataFrame(data={"patient":list(train_data_3.index), "hypothesis 1": list(m_3_tr_predictions_flat), 
                            "prediction":list(m_3_labeled_tr_predictions), "Platinum_Status":list(y_val_3)})
    tr_df_m_3.to_csv(save_prediction_path+"prediction_result_m_3_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

    ts_df_m_3 = pd.DataFrame(data={"patient":list(test_data_3.index), "hypothesis 1": list(m_3_ts_predictions_flat), 
                            "prediction":list(m_3_labeled_ts_predictions), "Platinum_Status":list(test_y_val_3)})
    ts_df_m_3.to_csv(save_prediction_path+"prediction_result_m_3_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])


    # ## 4) Model 4

    # In[ ]:


    print("Model_4: "+str(types[3]))

    # 1) parameter setting
    adam = optimizers.Adam(lr=0.01)
    input_drop_out_m_4 = 0.3
    drop_out_m_4 = 0.5
    layers = [5]
    m_4_tr_loss_best = 100 # for saving best loss value 
    best_m_4_model=[] #for saving best model
    count=0 # for early stopping

    # 2) model build
    input_m_4 = Input(shape=(features_4,))
    m_4_m_bn = Dropout(input_drop_out_m_4)(input_m_4)
    for i in layers:
        m_4_m = Dense(i)(m_4_m_bn)
        m_4_m_bn = BatchNormalization(axis=1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')(m_4_m)
        m_4_m_ac = Activation("relu")(m_4_m_bn)
    m_4_m_final = m_4_m_ac
    output_m_4 = Dense(1, activation="sigmoid")(m_4_m_final)
    m_4_model = Model(inputs=input_m_4,outputs=output_m_4)
    m_4_model.compile(optimizer=optimizers.Adam(), 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


    # 3) Training: if no increase of tr_loss three times, stop training.
    while 1:
        m_4_model.fit(x_val_4, y_val_4, batch_size=5, epochs=1)
        m_4_tr_loss, m_4_tr_accuracy =m_4_model.evaluate(x_val_4,y_val_4)
        if m_4_tr_loss < m_4_tr_loss_best: # new best model. count reset.
            m_4_tr_loss_best = m_4_tr_loss
            count=0
            best_m_4_model = m_4_model
            print("best model: "+str(m_4_tr_accuracy))
        if count>20: # no increase three time. stop.
            m_4_model = best_m_4_model
            break
        else: count=count+1
    print("Model_4 trained.")

    # 4) save model
    m_4_model.save(save_model_path+"/m_4.h5")
    print("Model_4 saved.")

    # 5) evaluate model
    m_4_output_list = model_performance(
        information = False, using_model=m_4_model,Input_Prediction_Passively = False, 
        tr_x_val=x_val_4, tr_y_val=y_val_4, ts_x_val=test_x_val_4, ts_y_val=test_y_val_4,
        output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                     "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                     "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                     "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                     "roc_auc_total"])

    m_4_tr_loss, m_4_tr_accuracy, m_4_tr_sensitivity, m_4_tr_specificity, m_4_tr_predictions, m_4_labeled_tr_predictions, m_4_tr_predictions_flat, m_4_roc_auc_tr, m_4_ts_loss, m_4_ts_accuracy, m_4_ts_sensitivity, m_4_ts_specificity, m_4_ts_predictions,m_4_labeled_ts_predictions, m_4_ts_predictions_flat, m_4_roc_auc_ts, m_4_roc_auc_total = m_4_output_list

    print("Overall AUC: ", m_4_roc_auc_total)
    print("Train AUC: ", m_4_roc_auc_tr)
    print("Test AUC: ", m_4_roc_auc_ts)

    print("Train Accuracy: {}".format(m_4_tr_accuracy))
    print("Train Sensitivities & Specificities : "+str(m_4_tr_sensitivity)+", "+str(m_4_tr_specificity))
    print("Test Accuracy: {}".format(m_4_ts_accuracy))
    print("Test Sensitivities & Specificities : "+str(m_4_ts_sensitivity)+", "+str(m_4_ts_specificity))


    # In[ ]:


    # save prediction result.

    tr_df_m_3 = pd.DataFrame(data={"patient":list(train_data_3.index), "hypothesis 1": list(m_3_tr_predictions_flat), 
                            "prediction":list(m_3_labeled_tr_predictions), "Platinum_Status":list(y_val_3)})
    tr_df_m_3.to_csv(save_prediction_path+"prediction_result_m_3_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

    ts_df_m_3 = pd.DataFrame(data={"patient":list(test_data_3.index), "hypothesis 1": list(m_3_ts_predictions_flat), 
                            "prediction":list(m_3_labeled_ts_predictions), "Platinum_Status":list(test_y_val_3)})
    ts_df_m_3.to_csv(save_prediction_path+"prediction_result_m_3_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])


    # ## 5) Model 5

    # In[ ]:


    print("Model_5: "+str(types[0]))

    # 1) parameter setting
    adam = optimizers.Adam(lr=0.01)
    input_drop_out_m_5 = 0.3
    drop_out_m_5 = 0.5
    layers = [5]
    m_5_tr_loss_best = 100 # for saving best loss value 
    best_m_5_model=[] #for saving best model
    count=0 # for early stopping

    # 2) model build
    input_m_5 = Input(shape=(features_5,))
    m_5_m_dp = Dropout(input_drop_out_m_5)(input_m_5)
    for i in layers:
        m_5_m = Dense(i,activation='relu')(m_5_m_dp)
        m_5_m_dp = Dropout(drop_out_m_5)(m_5_m)
    m_5_m_final = m_5_m_dp
    output_m_5 = Dense(1, activation="sigmoid")(m_5_m_final)
    m_5_model = Model(inputs=input_m_5,outputs=output_m_5)
    m_5_model.compile(optimizer=adam, 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


    # 3) Training: if no increase of tr_loss three times, stop training.
    while 1:
        m_5_model.fit(x_val_5, y_val_5, batch_size=5, nb_epoch=1)
        m_5_tr_loss=m_5_model.evaluate(x_val_5,y_val_5)[0]
        if m_5_tr_loss < m_5_tr_loss_best: # new best model. count reset.
            m_5_tr_loss_best = m_5_tr_loss
            count=0
            best_m_5_model = m_5_model
        if count>3: # no increase three time. stop.
            m_5_model = best_m_5_model
            break
        else: count=count+1
    print("Model_5 trained.")

    # 4) save model
    m_5_model.save(save_model_path+"/m_5.h5")
    print("Model_5 saved.")

    # 5) evaluate model
    m_5_output_list = model_performance(
        information = False, using_model=m_5_model,Input_Prediction_Passively = False, 
        tr_x_val=x_val_5, tr_y_val=y_val_5, ts_x_val=test_x_val_5, ts_y_val=test_y_val_5,
        output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                     "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                     "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                     "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                     "roc_auc_total"])

    m_5_tr_loss, m_5_tr_accuracy, m_5_tr_sensitivity, m_5_tr_specificity, m_5_tr_predictions, m_5_labeled_tr_predictions, m_5_tr_predictions_flat, m_5_roc_auc_tr, m_5_ts_loss, m_5_ts_accuracy, m_5_ts_sensitivity, m_5_ts_specificity, m_5_ts_predictions,m_5_labeled_ts_predictions, m_5_ts_predictions_flat, m_5_roc_auc_ts, m_5_roc_auc_total = m_5_output_list

    print("Overall AUC: ", m_5_roc_auc_total)
    print("Train AUC: ", m_5_roc_auc_tr)
    print("Test AUC: ", m_5_roc_auc_ts)

    print("Train Accuracy: {}".format(m_5_tr_accuracy))
    print("Train Sensitivities & Specificities : "+str(m_5_tr_sensitivity)+", "+str(m_5_tr_specificity))
    print("Test Accuracy: {}".format(m_5_ts_accuracy))
    print("Test Sensitivities & Specificities : "+str(m_5_ts_sensitivity)+", "+str(m_5_ts_specificity))


    # In[ ]:


    # save prediction result.

    tr_df_m_3 = pd.DataFrame(data={"patient":list(train_data_3.index), "hypothesis 1": list(m_3_tr_predictions_flat), 
                            "prediction":list(m_3_labeled_tr_predictions), "Platinum_Status":list(y_val_3)})
    tr_df_m_3.to_csv(save_prediction_path+"prediction_result_m_3_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

    ts_df_m_3 = pd.DataFrame(data={"patient":list(test_data_3.index), "hypothesis 1": list(m_3_ts_predictions_flat), 
                            "prediction":list(m_3_labeled_ts_predictions), "Platinum_Status":list(test_y_val_3)})
    ts_df_m_3.to_csv(save_prediction_path+"prediction_result_m_3_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])


    # ## 6) Model 6 

    # In[ ]:


    print("Model_6: "+str(types[0]))

    # 1) parameter setting
    adam = optimizers.Adam(lr=0.01)
    input_drop_out_m_6 = 0.3
    drop_out_m_6 = 0.5
    layers = [5]
    m_6_tr_loss_best = 100 # for saving best loss value 
    best_m_6_model=[] #for saving best model
    count=0 # for early stopping

    # 2) model build
    input_m_6 = Input(shape=(features_6,))
    m_6_m_dp = Dropout(input_drop_out_m_6)(input_m_6)
    for i in layers:
        m_6_m = Dense(i,activation='relu')(m_6_m_dp)
        m_6_m_dp = Dropout(drop_out_m_6)(m_6_m)
    m_6_m_final = m_6_m_dp
    output_m_6 = Dense(1, activation="sigmoid")(m_6_m_final)
    m_6_model = Model(inputs=input_m_6,outputs=output_m_6)
    m_6_model.compile(optimizer=adam, 
                    loss='binary_crossentropy',
                    metrics=['accuracy'])


    # 3) Training: if no increase of tr_loss three times, stop training.
    while 1:
        m_6_model.fit(x_val_6, y_val_6, batch_size=5, nb_epoch=1)
        m_6_tr_loss=m_6_model.evaluate(x_val_6,y_val_6)[0]
        if m_6_tr_loss < m_6_tr_loss_best: # new best model. count reset.
            m_6_tr_loss_best = m_6_tr_loss
            count=0
            best_m_6_model = m_6_model
        if count>3: # no increase three time. stop.
            m_6_model = best_m_6_model
            break
        else: count=count+1
    print("Model_6 trained.")

    # 4) save model
    m_6_model.save(save_model_path+"/m_6.h5")
    print("Model_6 saved.")

    # 5) evaluate model
    m_6_output_list = model_performance(
        information = False, using_model=m_6_model,Input_Prediction_Passively = False, 
        tr_x_val=x_val_6, tr_y_val=y_val_6, ts_x_val=test_x_val_6, ts_y_val=test_y_val_6,
        output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                     "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                     "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                     "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                     "roc_auc_total"])

    m_6_tr_loss, m_6_tr_accuracy, m_6_tr_sensitivity, m_6_tr_specificity, m_6_tr_predictions, m_6_labeled_tr_predictions, m_6_tr_predictions_flat, m_6_roc_auc_tr, m_6_ts_loss, m_6_ts_accuracy, m_6_ts_sensitivity, m_6_ts_specificity, m_6_ts_predictions,m_6_labeled_ts_predictions, m_6_ts_predictions_flat, m_6_roc_auc_ts, m_6_roc_auc_total = m_6_output_list

    print("Overall AUC: ", m_6_roc_auc_total)
    print("Train AUC: ", m_6_roc_auc_tr)
    print("Test AUC: ", m_6_roc_auc_ts)

    print("Train Accuracy: {}".format(m_6_tr_accuracy))
    print("Train Sensitivities & Specificities : "+str(m_6_tr_sensitivity)+", "+str(m_6_tr_specificity))
    print("Test Accuracy: {}".format(m_6_ts_accuracy))
    print("Test Sensitivities & Specificities : "+str(m_6_ts_sensitivity)+", "+str(m_6_ts_specificity))


    # In[ ]:


    # save prediction result.

    tr_df_m_6 = pd.DataFrame(data={"patient":list(train_data_6.index), "hypothesis 1": list(m_6_tr_predictions_flat), 
                            "prediction":list(m_6_labeled_tr_predictions), "Platinum_Status":list(y_val_6)})
    tr_df_m_6.to_csv(save_prediction_path+"prediction_result_m_6_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

    ts_df_m_6 = pd.DataFrame(data={"patient":list(test_data_6.index), "hypothesis 1": list(m_6_ts_predictions_flat), 
                            "prediction":list(m_6_labeled_ts_predictions), "Platinum_Status":list(test_y_val_6)})
    ts_df_m_6.to_csv(save_prediction_path+"prediction_result_m_6_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])


    # ## Performance Comparison

    # In[ ]:


    label = []
    tr_accuracy_list = [m_1_tr_accuracy, m_2_tr_accuracy, m_3_tr_accuracy, m_4_tr_accuracy, m_5_tr_accuracy, m_6_tr_accuracy]
    ts_accuracy_list = [m_1_ts_accuracy, m_2_ts_accuracy, m_3_ts_accuracy, m_4_ts_accuracy, m_5_ts_accuracy, m_6_ts_accuracy]

    for i in range(1,7):
        label.append("model"+str(i))

    for model_num in range(len(label)):
        print("< "+label[model_num]+" > tr: "+str(tr_accuracy_list[model_num])+", ts: "+str(ts_accuracy_list[model_num]))


    # In[ ]:


    def plot_bar_x():
        # this is for plotting purpose
        plt.figure(figsize=(30,20))
        axes = plt.gca()
        axes.set_ylim([min(ts_accuracy_list)-0.02, max(ts_accuracy_list)+0.02])
        index = np.arange(len(label))
        plt.bar(index, ts_accuracy_list,color=['red', 'orange', 'yellow', "green",'blue', 'purple'],alpha=0.5,width=0.3)
        plt.xlabel('Method', fontsize=35)
        plt.ylabel('Accuracy', fontsize=35)
        plt.yticks(fontsize=30)    
        plt.xticks(index, types, fontsize=30, rotation=90)
        plt.title('Performance Comparison for each Models',fontsize=40)
        plt.show()


    # In[ ]:


    plot_bar_x()


    # In[ ]:





    # In[ ]:




