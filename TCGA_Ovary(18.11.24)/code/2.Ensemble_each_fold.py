
# coding: utf-8

# # Importing Library

# In[13]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Input, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import Model, load_model, Sequential 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

early_stopping = EarlyStopping(patience=10)

# Helper libraries
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
from os import listdir

np.random.seed(777)

print(tf.__version__)


# ##  Functions library

# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


# Coverage algorithm
def ensemble_coverage(inputModels,x,y):
    
    outputModels = []
    modelInfo = []
    coverageTotal= [False]*len(y)
    
    for i in range(len(inputModels)):
        m = inputModels[i]
        yHat = m.predict(x[i])
        yHat = [round(i) for [i] in yHat]
        
        loss, acc = m.evaluate(x[i],y)
        modelInfo.append((m,yHat,acc))
        #print(yHat[0:10])
        #print(acc)
    
    modelInfo.sort(key=lambda x : x[2],reverse=True)
    #print(modelInfo)
    
    for m,yHat,acc in modelInfo:
        beforeCoverage = sum(coverageTotal)
        coverage = [a == b for a,b in zip(y,yHat)]
        coverageTotal = [a or b for a,b in zip(coverageTotal,coverage)]
        afterCoverage = sum(coverageTotal)
        
        print(afterCoverage/len(y))
        
        if afterCoverage > beforeCoverage:
            outputModels.append(m)
            print("Increased Coverage : model added!")
        else:
            print("Same Coverage : model not added")
        if afterCoverage == len(y):
            print("Fully Covered!")
            break
                
    return outputModels


# # 1. Preparation: import & preprocessing data + import module

# ## Input path & name of models / raw data for ensemble

# In[18]:


# change model_path & each model_name.

types = ["OV_six_fold_Annotation3000_400", 
         "OV_six_fold_CV_400", 
         "OV_six_fold_Var_400", 
         "OV_six_fold_new_Diff_400",
         "OV_six_fold_Clin", 
         "OV_six_fold_SNV_400" 
         ]

ch = input("test index(1~5): ")
ts_i = int(ch)

# input model path & ensemble data(Transcriptome, Cinical Information, Somatic Mutation data)
# data path(server): /home/tjahn/TCGA_Ovary/01.Data/DNN/TC_intersect_subsamples_by_names 
input_model_path = "../best_models/test_"+str(ts_i)+"/"
path = "../TC_six_fold_subsamples/"
save_model_path = "../garbage/"
save_prediction_path = "../garbage/"
save_result_path = "../garbage/"

model_names = []
model_index = []
files = os.listdir(input_model_path)

for f in files:
    ext= os.path.splitext(f)[-1]
    if ext == ".h5":
        model_names.append(os.path.splitext(f)[0])
        ind = int(f.split("_")[1].split("-")[0])
        model_index.append(ind)


# In[19]:


for t in range(len(model_index)):
    print(model_names[t])
    print(types[model_index[t]]+"\n")
    t = t+1


# ## Import Data

# In[20]:


file_1 = path+types[0]+".csv"
file_2 = path+types[1]+".csv"
file_3 = path+types[2]+".csv"
file_4 = path+types[3]+".csv"
file_5 = path+types[4]+".csv"
file_6 = path+types[5]+".csv"

idx_col = 0

full_data_1 = pd.read_csv(file_1,index_col=idx_col)
full_data_2 = pd.read_csv(file_2,index_col=idx_col)
full_data_3 = pd.read_csv(file_3,index_col=idx_col)
full_data_4 = pd.read_csv(file_4,index_col=idx_col)
full_data_5 = pd.read_csv(file_5,index_col=idx_col)
full_data_6 = pd.read_csv(file_6,index_col=idx_col)

inter_data_1 = full_data_1.iloc[list(full_data_1.iloc[:,-1]!=6)]
inter_data_2 = full_data_2.iloc[list(full_data_2.iloc[:,-1]!=6)]
inter_data_3 = full_data_3.iloc[list(full_data_3.iloc[:,-1]!=6)]
inter_data_4 = full_data_4.iloc[list(full_data_4.iloc[:,-1]!=6)]
inter_data_5 = full_data_5.iloc[list(full_data_5.iloc[:,-1]!=6)]
inter_data_6 = full_data_6.iloc[list(full_data_6.iloc[:,-1]!=6)]

full_ds_list = [full_data_1, full_data_2, full_data_3, full_data_4, full_data_5, full_data_6]
inter_ds_list = [inter_data_1, inter_data_2, inter_data_3, inter_data_4, inter_data_5, inter_data_6]

# Split Train Test Data & Make full & inter dataset

full_dataset = {"tr_data":[], "ts_data":[], "tr_y_val":[], "tr_x_val":[], "ts_y_val":[], "ts_x_val":[]}
inter_dataset = {"tr_data":[], "ts_data":[], "tr_y_val":[], "tr_x_val":[], "ts_y_val":[], "ts_x_val":[]}


print("############### test index is ["+str(ts_i)+"] ###############\n\n")
for m in range(len(model_index)):
    print("model index: "+str(model_index[m]))
    full_tr_data, full_ts_data, full_tr_y_val, full_tr_x_val, full_ts_y_val, full_ts_x_val = data_split(raw_data = full_ds_list[model_index[m]], index_col = -1, test_index = ts_i)
    print("["+str(m)+"]: "+model_names[m]+" for type: "+types[model_index[m]]+".\n full tr & ts: "+str(full_tr_x_val.shape)+", "+str(full_ts_x_val.shape)+"\n")
    full_dataset['tr_data'].append(full_tr_data)
    full_dataset['ts_data'].append(full_ts_data)
    full_dataset['tr_x_val'].append(full_tr_x_val)
    full_dataset['tr_y_val'].append(full_tr_y_val)
    full_dataset['ts_x_val'].append(full_ts_x_val)
    full_dataset['ts_y_val'].append(full_ts_y_val)  
    inter_tr_data, inter_ts_data, inter_tr_y_val, inter_tr_x_val, inter_ts_y_val, inter_ts_x_val = data_split(raw_data = inter_ds_list[model_index[m]], index_col = -1, test_index = ts_i)
    inter_dataset['tr_data'].append(inter_tr_data)
    inter_dataset['ts_data'].append(inter_ts_data)
    inter_dataset['tr_x_val'].append(inter_tr_x_val)
    inter_dataset['tr_y_val'].append(inter_tr_y_val)
    inter_dataset['ts_x_val'].append(inter_ts_x_val)
    inter_dataset['ts_y_val'].append(inter_ts_y_val)     


# In[21]:


inter_newDiff_dataset = {"tr_data":[], "ts_data":[], "tr_y_val":[], "tr_x_val":[], "ts_y_val":[], "ts_x_val":[]}
inter_tr_data, inter_ts_data, inter_tr_y_val, inter_tr_x_val, inter_ts_y_val, inter_ts_x_val = data_split(raw_data = inter_ds_list[3], index_col = -1, test_index = ts_i)
inter_newDiff_dataset['tr_data']= inter_tr_data
inter_newDiff_dataset['ts_data']= inter_ts_data
inter_newDiff_dataset['tr_x_val']= inter_tr_x_val
inter_newDiff_dataset['tr_y_val']= inter_tr_y_val
inter_newDiff_dataset['ts_x_val']= inter_ts_x_val
inter_newDiff_dataset['ts_y_val']= inter_ts_y_val    
#inter_new_Diff_dataset = inter_dataset[1]


# ## Import separate models & evaluation

# In[22]:


# model load & evaluation. <model_n_l> is full-layer model, <model_n_l_new> is without-sigmoid-layer model.
'''
Each model's tr_accuracy can be differ to original model, but ts_accuracy should be same to original tested models.
Because we using full-size data(about 200 patients data used Transcriptome, Clinical, SNV models.) for train each models.
In contrast, in this code, we using ensemble-input data(intersected 153 patients).
For-training-patients may be different in ensemble data and whole size data, but for-test-patients are the same.
'''

model_list = []
model_output_list = {"tr_accuracy":[], "tr_sensitivity":[], "tr_specificity":[], "tr_predictions":[],
                 "labeled_tr_predictions":[], "tr_predictions_flat":[], "roc_auc_tr":[], 
                 "ts_accuracy":[], "ts_sensitivity":[], "ts_specificity":[], "ts_predictions":[],
                 "labeled_ts_predictions":[], "ts_predictions_flat":[], "roc_auc_ts":[], 
                 "roc_auc_total":[], "tr_result":[], "ts_result":[]}
tr_predictions = []
ts_predictions = []

for m in range(len(model_names)):
    
    model_l = load_model(input_model_path+model_names[m]+".h5")
    model_list.append(model_l)
    output_list = output_list = model_performance(
        information = False, using_model=model_l,Input_Prediction_Passively = False, 
        tr_x_val=inter_dataset['tr_x_val'][m], tr_y_val=inter_dataset['tr_y_val'][m], ts_x_val=inter_dataset['ts_x_val'][m], ts_y_val=inter_dataset['ts_y_val'][m],
        output_list=["tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                     "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                     "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                     "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                     "roc_auc_total"])
    m_tr_accuracy, m_tr_sensitivity, m_tr_specificity, m_tr_predictions, m_labeled_tr_predictions, m_tr_predictions_flat, m_roc_auc_tr, m_ts_accuracy, m_ts_sensitivity, m_ts_specificity, m_ts_predictions,m_labeled_ts_predictions, m_ts_predictions_flat, m_roc_auc_ts, m_roc_auc_total = output_list
    print("\nmodel: "+model_names[m])
    print("tr & ts for inter data: "+str(m_tr_accuracy)+", "+str(m_ts_accuracy)+"\n")
    
    model_l_new = Model(inputs = model_l.input, outputs=model_l.get_layer(model_l.layers[-2].name).output)
    m_tr_result = model_l_new.predict([inter_dataset['tr_x_val'][m]])
    m_ts_result = model_l_new.predict([inter_dataset['ts_x_val'][m]])
    
    model_output_list["tr_accuracy"].append(m_tr_accuracy)
    model_output_list["tr_sensitivity"].append(m_tr_sensitivity)
    model_output_list["tr_specificity"].append(m_tr_specificity)
    model_output_list["ts_accuracy"].append(m_ts_accuracy)
    model_output_list["ts_sensitivity"].append(m_ts_sensitivity)
    model_output_list["ts_specificity"].append(m_ts_specificity)
    model_output_list["tr_result"].append(m_tr_result)
    
    model_output_list["tr_predictions"].append(m_tr_predictions)
    model_output_list["labeled_tr_predictions"].append(m_labeled_tr_predictions)
    model_output_list["tr_predictions_flat"].append(m_tr_predictions_flat)
    model_output_list["roc_auc_tr"].append(m_roc_auc_tr)
    model_output_list["ts_predictions"].append(m_ts_predictions)
    model_output_list["labeled_ts_predictions"].append(m_labeled_ts_predictions)
    model_output_list["ts_predictions_flat"].append(m_ts_predictions_flat)
    model_output_list["roc_auc_ts"].append(m_roc_auc_ts)
    model_output_list["ts_result"].append(m_ts_result)
    
    model_output_list["roc_auc_total"].append(m_roc_auc_total)  


# In[25]:


print(model_output_list['ts_accuracy'][0], model_output_list['ts_sensitivity'][0], model_output_list['ts_specificity'][0])


# ### Ensemble models selection

# In[13]:


# select models automatically

e_models_select = ensemble_coverage(model_list,inter_dataset["tr_x_val"],inter_dataset["tr_y_val"][0])
print("model numbers: "+str(len(e_models_select)))


# In[16]:


# select models manually

for m in range(len(model_names)):
    print("#### ["+str(m+1)+"] "+model_names[m]+" ####")
    print("types: "+types[model_index[m]])
    print("tr: "+str(model_output_list["tr_accuracy"][m])+", ts: "+str(model_output_list["ts_accuracy"][m])+"\n")

comb = ''
select = []
while 1:
    ch = input("input numbers for selection(1 ~ "+str(len(model_names))+". \'q\' for quit.: ")
    if(ch == 'q'):
        break
    else:
        select.append(int(ch))
        if(len(comb)<1):
            comb = model_names[int(ch)-1]
        else:
            comb = comb + ", " +model_names[int(ch)-1]
        
print("################### select models: "+comb+" ###################")


# ## variables for saving performance & hyperparameters

# In[17]:


model_type_list = []
model_comb_list = []
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


k = 0
lr_box = []
layers_box = []
batch_size_box = []
drop_out_box = []
input_drop_out_box = []
batch_normalize_box = []
count_lim_box=[]


# # 2. Modeling Ensemble model

# ## 1) DNN-Combiner Ensmeble

# ### Ensemble Input listup

# In[18]:


m_tr_predictions_select = []
m_ts_predictions_select = []   

for i in range(len(select)):
    m_tr_predictions_select.append(model_output_list["tr_predictions"][select[i]-1])
    m_ts_predictions_select.append(model_output_list["ts_predictions"][select[i]-1])
    #print(m_tr_predictions[select[i]-1].shape)
    
em_tr_x_val = np.concatenate(m_tr_predictions_select, axis=1)
em_ts_x_val = np.concatenate(m_ts_predictions_select, axis=1)

tr_y_val = inter_dataset["tr_y_val"][0]
ts_y_val = inter_dataset["ts_y_val"][0]


# In[19]:


print(em_tr_x_val.shape)
print(em_ts_x_val.shape)


# In[20]:


# parameter change

lr = 0.01
drop_out_m = 0
input_drop_out_m = 0.3
batch_size = 10
BN = True
layers = [3]
count_lim = 10


# In[22]:


print("################################## DNN em ##################################")
print("select: "+str(select))
for select_i in select:
    print("\n"+types[model_index[select_i-1]])
    print(model_names[select_i-1])

    
print("#############################################################################################")

# 1) parameter setting
em_adam = optimizers.Adam(lr=lr)                                   
em_tr_loss_best = 100 # for saving best loss value 
em_best_model=[] #for saving best model
count=0 # for early stopping

# 2) model build
em_input = Input(shape=(len(select),))
em_dp = Dropout(input_drop_out_m)(em_input)
for l in layers:
    if BN == True:
        em_m = Dense(l)(em_dp)
        em_bn = BatchNormalization(axis=1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')(em_m)
        em_dp = Activation("relu")(em_bn)
    else:
        em_m = Dense(l,activation='relu')(em_dp)
        em_dp = Dropout(drop_out_m)(em_m)

em_final = em_dp
em_output = Dense(1, activation="sigmoid")(em_final)
em_model = Model(inputs=em_input,outputs=em_output)
em_model.compile(optimizer=em_adam, 
                loss='binary_crossentropy',
                metrics=['accuracy'])

# 3) Training: if no increase of tr_loss three times, stop training.
while 1:
    em_model.fit(em_tr_x_val, tr_y_val, batch_size=batch_size, nb_epoch=1, verbose = 0)
    em_tr_loss=em_model.evaluate( em_tr_x_val, tr_y_val)[0]
    if em_tr_loss < em_tr_loss_best: # new best model. count reset.
        em_tr_loss_best = em_tr_loss
        count=0
        em_best_model = em_model
    if count>count_lim: # no increase three time. stop.
        em_model = em_best_model
        break
    else: count=count+1
print("Model emDNN" +"-"+str(ts_i)+" trained.")

# 4) save model
em_model.save(save_model_path+"/m_emDNN-"+str(ts_i)+"_"+str(k)+".h5")


# ### Evaluating _DNN Combiner_ ensemble model

# In[23]:


em_output_list = model_performance(
    information = False, using_model=em_model,Input_Prediction_Passively = False, 
    tr_x_val=em_tr_x_val, tr_y_val=tr_y_val, ts_x_val=em_ts_x_val, ts_y_val=ts_y_val,
    output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                 "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                 "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                 "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                 "roc_auc_total"])

em_tr_loss, em_tr_accuracy, em_tr_sensitivity, em_tr_specificity, em_tr_predictions, em_labeled_tr_predictions, em_tr_predictions_flat, em_roc_auc_tr, em_ts_loss, em_ts_accuracy, em_ts_sensitivity, em_ts_specificity, em_ts_predictions,em_labeled_ts_predictions, em_ts_predictions_flat, em_roc_auc_ts, em_roc_auc_total = em_output_list

print("Overall AUC: ", em_roc_auc_total)
print("Train AUC: ", em_roc_auc_tr)
print("Test AUC: ", em_roc_auc_ts)

print("Train Accuracy: {}".format(em_tr_accuracy))
print("Train Sensitivities & Specificities : "+str(em_tr_sensitivity)+", "+str(em_tr_specificity))
print("Test Accuracy: {}".format(em_ts_accuracy))
print("Test Sensitivities & Specificities : "+str(em_ts_sensitivity)+", "+str(em_ts_specificity))

# save prediction result.

tr_df_em = pd.DataFrame(data={"patient":list(inter_dataset["tr_data"][0].index), "hypothesis 1": list(em_tr_predictions_flat), 
                        "prediction":list(em_labeled_tr_predictions), "Platinum_Status":list(tr_y_val)})
tr_df_em.to_csv(save_prediction_path+"emDNN-"+str(ts_i)+"_"+str(k)+"_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

ts_df_em = pd.DataFrame(data={"patient":list(inter_dataset["ts_data"][0].index), "hypothesis 1": list(em_ts_predictions_flat), 
                        "prediction":list(em_labeled_ts_predictions), "Platinum_Status":list(ts_y_val)})
ts_df_em.to_csv(save_prediction_path+"emDNN-"+str(ts_i)+"_"+str(k)+"_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

count_lim_box.append(count_lim)
lr_box.append(lr)
layers_box.append(layers)
batch_size_box.append(batch_size)
drop_out_box.append(drop_out_m)
input_drop_out_box.append(input_drop_out_m)
batch_normalize_box.append(BN)
model_type_list.append("em-DNN")
model_comb_list.append(comb)
test_index_list.append(ts_i)
tr_acc_list.append(em_tr_accuracy)
ts_acc_list.append(em_ts_accuracy)
tr_sensitivity_list.append(em_tr_sensitivity)
ts_sensitivity_list.append(em_ts_sensitivity)
tr_specificity_list.append(em_tr_specificity)
ts_specificity_list.append(em_ts_specificity)
tr_auc_list.append(em_roc_auc_tr)
ts_auc_list.append(em_roc_auc_ts)
tot_auc_list.append(em_roc_auc_total)

k = k+1


# ## 2) Mean Ensemble

# ### Evaluating _mean_ ensemble model

# In[24]:


mean_em_tr_predictions=sum(m_tr_predictions_select)/len(select)
mean_em_ts_predictions=sum(m_ts_predictions_select)/len(select)

mean_em_output_list = model_performance(
    information = False, using_model=None,Input_Prediction_Passively = True, 
    tr_predictions=mean_em_tr_predictions, ts_predictions=mean_em_ts_predictions, 
    tr_x_val=em_tr_x_val, tr_y_val=tr_y_val, ts_x_val=em_ts_x_val, ts_y_val=ts_y_val,
    output_list=["tr_sensitivity", "tr_specificity",
                 "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                 "ts_sensitivity", "ts_specificity",
                 "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                 "roc_auc_total"])
mean_em_tr_sensitivity, mean_em_tr_specificity,  mean_em_labeled_tr_predictions, mean_em_tr_predictions_flat, mean_em_roc_auc_tr, mean_em_ts_sensitivity, mean_em_ts_specificity, mean_em_labeled_ts_predictions, mean_em_ts_predictions_flat, mean_em_roc_auc_ts, mean_em_roc_auc_total = mean_em_output_list

mean_em_tr_accuracy = sum(mean_em_labeled_tr_predictions==tr_y_val.values)/len(tr_y_val)
mean_em_ts_accuracy = sum(mean_em_labeled_ts_predictions==ts_y_val.values)/len(ts_y_val)

print("Overall AUC: ", mean_em_roc_auc_total)
print("Train AUC: ", mean_em_roc_auc_tr)
print("Test AUC: ", mean_em_roc_auc_ts)

print("Train Accuracy: {}".format(mean_em_tr_accuracy))
print("Train Sensitivities & Specificities : "+str(mean_em_tr_sensitivity)+", "+str(mean_em_tr_specificity))
print("Test Accuracy: {}".format(mean_em_ts_accuracy))
print("Test Sensitivities & Specificities : "+str(mean_em_ts_sensitivity)+", "+str(mean_em_ts_specificity))

# save prediction result.

tr_df_mean = pd.DataFrame(data={"patient":list(inter_dataset["tr_data"][0].index), "hypothesis 1": list(mean_em_tr_predictions_flat), 
                        "prediction":list(mean_em_labeled_tr_predictions), "Platinum_Status":list(tr_y_val)})
tr_df_mean.to_csv(save_prediction_path+"emMean-"+str(ts_i)+"_"+str(k)+"_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

ts_df_mean = pd.DataFrame(data={"patient":list(inter_dataset["ts_data"][0].index), "hypothesis 1": list(mean_em_ts_predictions_flat), 
                        "prediction":list(mean_em_labeled_ts_predictions), "Platinum_Status":list(ts_y_val)})
ts_df_mean.to_csv(save_prediction_path+"emMean-"+str(ts_i)+"_"+str(k)+"_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

count_lim_box.append("-")
lr_box.append("-")
layers_box.append("-")
batch_size_box.append("-")
drop_out_box.append("-")
input_drop_out_box.append("-")
batch_normalize_box.append("-")
model_type_list.append("em-Mean")
model_comb_list.append(comb)
test_index_list.append(ts_i)
tr_acc_list.append(mean_em_tr_accuracy)
ts_acc_list.append(mean_em_ts_accuracy)
tr_sensitivity_list.append(mean_em_tr_sensitivity)
ts_sensitivity_list.append(mean_em_ts_sensitivity)
tr_specificity_list.append(mean_em_tr_specificity)
ts_specificity_list.append(mean_em_ts_specificity)
tr_auc_list.append(mean_em_roc_auc_tr)
ts_auc_list.append(mean_em_roc_auc_ts)
tot_auc_list.append(mean_em_roc_auc_total)

k = k+1


# ## 3) Transferred Ensemble Modeling 

# ### Making new input data for t-ensemble

# In[25]:


m_tr_result_select = []
m_ts_result_select = []

for i in range(len(select)):
    m_tr_result_select.append(model_output_list["tr_result"][select[i]-1])
    m_ts_result_select.append(model_output_list["ts_result"][select[i]-1])

t_em_tr_x_val = np.concatenate(m_tr_result_select, axis=1)
t_em_ts_x_val = np.concatenate(m_ts_result_select, axis=1)
print("\n############################################### t-em x val merged. ###############################################\n")
print(t_em_tr_x_val.shape)
print(t_em_ts_x_val.shape)


# ### Modeling t-ensemble  

# In[26]:


# parameter change

lr = 0.01
drop_out_m = 0
input_drop_out_m = 0.2
batch_size = 5
BN = True
layers = [100]
count_lim = 10


# In[27]:


print("################################## Transferred em ##################################")
print("select: "+str(select))
for select_i in select:
    print("\n"+types[model_index[select_i-1]])
    print(model_names[select_i-1])

    
print("#############################################################################################")

# 1) parameter setting
t_em_adam = optimizers.Adam(lr=lr)                                   
input_drop_out_m = 0.3
drop_out_m = 0
batch_size = 5
BN = True                           
layers = [100]

t_em_tr_loss_best = 100 # for saving best loss value 
t_em_best_model=[] #for saving best model
count=0 # for early stopping

# 2) model build
t_em_input = Input(shape=(t_em_ts_x_val.shape[1],))
t_em_dp = Dropout(input_drop_out_m)(t_em_input)
for l in layers:
    if BN == True:
        t_em_m = Dense(l)(t_em_dp)
        t_em_bn = BatchNormalization(axis=1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')(t_em_m)
        t_em_dp = Activation("relu")(t_em_bn)
    else:
        t_em_m = Dense(l,activation='relu')(t_em_dp)
        t_em_dp = Dropout(drop_out_m)(t_em_m)

t_em_final = t_em_dp
t_em_output = Dense(1, activation="sigmoid")(t_em_final)
t_em_model = Model(inputs=t_em_input,outputs=t_em_output)
t_em_model.compile(optimizer=t_em_adam, 
                loss='binary_crossentropy',
                metrics=['accuracy'])

# 3) Training: if no increase of tr_loss three times, stop training.
while 1:
    t_em_model.fit(t_em_tr_x_val, tr_y_val, batch_size=batch_size, nb_epoch=1, verbose = 0)
    t_em_tr_loss=t_em_model.evaluate( t_em_tr_x_val, tr_y_val)[0]
    if t_em_tr_loss < t_em_tr_loss_best: # new best model. count reset.
        t_em_tr_loss_best = t_em_tr_loss
        count=0
        t_em_best_model = t_em_model
    if count>count_lim: # no increase three time. stop.
        t_em_model = t_em_best_model
        break
    else: count=count+1
        
print("Model em_T" +"-"+str(ts_i)+" trained.")

# 4) save model
em_model.save(save_model_path+"/m_emT-"+str(ts_i)+"_"+str(k)+".h5")


# ### Evaluating t-ensemble

# In[28]:


t_em_output_list = model_performance(
    information = False, using_model=t_em_model,Input_Prediction_Passively = False, 
    tr_x_val=t_em_tr_x_val, tr_y_val=tr_y_val, ts_x_val=t_em_ts_x_val, ts_y_val=ts_y_val,
    output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                 "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                 "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                 "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                 "roc_auc_total"])

t_em_tr_loss, t_em_tr_accuracy, t_em_tr_sensitivity, t_em_tr_specificity, t_em_tr_predictions, t_em_labeled_tr_predictions, t_em_tr_predictions_flat, t_em_roc_auc_tr, t_em_ts_loss, t_em_ts_accuracy, t_em_ts_sensitivity, t_em_ts_specificity, t_em_ts_predictions,t_em_labeled_ts_predictions, t_em_ts_predictions_flat, t_em_roc_auc_ts, t_em_roc_auc_total = t_em_output_list

print("Overall AUC: ", t_em_roc_auc_total)
print("Train AUC: ", t_em_roc_auc_tr)
print("Test AUC: ", t_em_roc_auc_ts)

print("Train Accuracy: {}".format(t_em_tr_accuracy))
print("Train Sensitivities & Specificities : "+str(t_em_tr_sensitivity)+", "+str(t_em_tr_specificity))
print("Test Accuracy: {}".format(t_em_ts_accuracy))
print("Test Sensitivities & Specificities : "+str(t_em_ts_sensitivity)+", "+str(t_em_ts_specificity))

# save prediction result.

tr_df_t_em = pd.DataFrame(data={"patient":list(inter_dataset["tr_data"][0].index), "hypothesis 1": list(t_em_tr_predictions_flat), 
                        "prediction":list(t_em_labeled_tr_predictions), "Platinum_Status":list(tr_y_val)})
tr_df_t_em.to_csv(save_prediction_path+"m_emT-"+str(ts_i)+"_"+str(k)+"_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

ts_df_t_em = pd.DataFrame(data={"patient":list(inter_dataset["ts_data"][0].index), "hypothesis 1": list(t_em_ts_predictions_flat), 
                        "prediction":list(t_em_labeled_ts_predictions), "Platinum_Status":list(ts_y_val)})
ts_df_t_em.to_csv(save_prediction_path+"m_emT-"+str(ts_i)+"_"+str(k)+"_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

count_lim_box.append(count_lim)
lr_box.append(lr)
layers_box.append(layers)
batch_size_box.append(batch_size)
drop_out_box.append(drop_out_m)
input_drop_out_box.append(input_drop_out_m)
batch_normalize_box.append(BN)
model_type_list.append("em-T")
model_comb_list.append(comb)
test_index_list.append(ts_i)
tr_acc_list.append(t_em_tr_accuracy)
ts_acc_list.append(t_em_ts_accuracy)
tr_sensitivity_list.append(t_em_tr_sensitivity)
ts_sensitivity_list.append(t_em_ts_sensitivity)
tr_specificity_list.append(t_em_tr_specificity)
ts_specificity_list.append(t_em_ts_specificity)
tr_auc_list.append(t_em_roc_auc_tr)
ts_auc_list.append(t_em_roc_auc_ts)
tot_auc_list.append(t_em_roc_auc_total)

k = k+1


# ## Hybrid Ensemble

# ### hybrid ensemble input dataset

# In[29]:


# dataset : raw data + prediction results
h_em_tr_x_val = np.concatenate([inter_newDiff_dataset["tr_x_val"], em_tr_x_val], axis = 1)
h_em_ts_x_val = np.concatenate([inter_newDiff_dataset["ts_x_val"], em_ts_x_val], axis = 1)


# In[30]:


print(h_em_tr_x_val.shape)
print(h_em_ts_x_val.shape)


# In[31]:


# parameter change

lr = 0.05
drop_out_m = 0
input_drop_out_m = 0.4
batch_size = 7
BN = True
layers = [120]
count_lim = 15


# In[32]:


print("hybrid ensemble model")

# 1) parameter setting
h_em_adam = optimizers.Adam(lr=lr)
h_em_tr_loss_best = 100 # for saving best loss value 
h_em_best_model=[] #for saving best model
count=0 # for early stopping

# 2) model build
h_em_input = Input(shape=(h_em_ts_x_val.shape[1],))
h_em_dp = Dropout(input_drop_out_m)(h_em_input)
for l in layers:
    if BN == True:
        h_em_m = Dense(l)(h_em_dp)
        h_em_bn = BatchNormalization(axis=1, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones')(h_em_m)
        h_em_dp = Activation("relu")(h_em_bn)
    else:
        h_em_m = Dense(l,activation='relu')(h_em_dp)
        h_em_dp = Dropout(drop_out_m)(h_em_m)

h_em_final = h_em_dp
h_em_output = Dense(1, activation="sigmoid")(h_em_final)
h_em_model = Model(inputs=h_em_input,outputs=h_em_output)
h_em_model.compile(optimizer=h_em_adam, 
                loss='binary_crossentropy',
                metrics=['accuracy'])

# 3) Training: if no increase of tr_loss three times, stop training.
while 1:
    h_em_model.fit(h_em_tr_x_val, tr_y_val, batch_size=batch_size, nb_epoch=1, verbose = 0)
    h_em_tr_loss=h_em_model.evaluate( h_em_tr_x_val, tr_y_val)[0]
    if h_em_tr_loss < h_em_tr_loss_best: # new best model. count reset.
        h_em_tr_loss_best = h_em_tr_loss
        count=0
        h_em_best_model = h_em_model
    if count>count_lim: # no increase three time. stop.
        h_em_model = h_em_best_model
        break
    else: count=count+1
        
print("Model h-em" +"-"+str(ts_i)+" trained.")

# 4) save model
em_model.save(save_model_path+"/m_emH-"+str(ts_i)+".h5")

# 5) evaluate model
h_em_output_list = model_performance(
    information = False, using_model=h_em_model,Input_Prediction_Passively = False, 
    tr_x_val=h_em_tr_x_val, tr_y_val=tr_y_val, ts_x_val=h_em_ts_x_val, ts_y_val=ts_y_val,
    output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                 "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                 "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                 "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                 "roc_auc_total"])

h_em_tr_loss, h_em_tr_accuracy, h_em_tr_sensitivity, h_em_tr_specificity, h_em_tr_predictions, h_em_labeled_tr_predictions, h_em_tr_predictions_flat, h_em_roc_auc_tr, h_em_ts_loss, h_em_ts_accuracy, h_em_ts_sensitivity, h_em_ts_specificity, h_em_ts_predictions,h_em_labeled_ts_predictions, h_em_ts_predictions_flat, h_em_roc_auc_ts, h_em_roc_auc_total = h_em_output_list

print("Overall AUC: ", h_em_roc_auc_total)
print("Train AUC: ", h_em_roc_auc_tr)
print("Test AUC: ", h_em_roc_auc_ts)

print("Train Accuracy: {}".format(h_em_tr_accuracy))
print("Train Sensitivities & Specificities : "+str(h_em_tr_sensitivity)+", "+str(h_em_tr_specificity))
print("Test Accuracy: {}".format(h_em_ts_accuracy))
print("Test Sensitivities & Specificities : "+str(h_em_ts_sensitivity)+", "+str(h_em_ts_specificity))

# save prediction result.

tr_df_h_em = pd.DataFrame(data={"patient":list(inter_dataset["tr_data"][0].index), "hypothesis 1": list(h_em_tr_predictions_flat), 
                        "prediction":list(h_em_labeled_tr_predictions), "Platinum_Status":list(tr_y_val)})
tr_df_h_em.to_csv(save_prediction_path+"m_emH-"+str(ts_i)+"_"+str(k)+"_tr.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

ts_df_h_em = pd.DataFrame(data={"patient":list(inter_dataset["ts_data"][0].index), "hypothesis 1": list(h_em_ts_predictions_flat), 
                        "prediction":list(h_em_labeled_ts_predictions), "Platinum_Status":list(ts_y_val)})
ts_df_h_em.to_csv(save_prediction_path+"m_emH-"+str(ts_i)+"_"+str(k)+"_ts.csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

count_lim_box.append(count_lim)
lr_box.append(lr)
layers_box.append(layers)
batch_size_box.append(batch_size)
drop_out_box.append(drop_out_m)
input_drop_out_box.append(input_drop_out_m)
batch_normalize_box.append(BN)
model_type_list.append("em-H")
model_comb_list.append(comb)
test_index_list.append(ts_i)
tr_acc_list.append(h_em_tr_accuracy)
ts_acc_list.append(h_em_ts_accuracy)
tr_sensitivity_list.append(h_em_tr_sensitivity)
ts_sensitivity_list.append(h_em_ts_sensitivity)
tr_specificity_list.append(h_em_tr_specificity)
ts_specificity_list.append(h_em_ts_specificity)
tr_auc_list.append(h_em_roc_auc_tr)
ts_auc_list.append(h_em_roc_auc_ts)
tot_auc_list.append(h_em_roc_auc_total)

k = k+1


# In[33]:


df_1 = df(data = {'model_types':model_type_list,
                  'model_comb':model_comb_list,
                'index': range(0,k),
                'rate':lr_box,
                'count':count_lim_box,
                'layers':layers_box,
                'batch_size': batch_size_box,
                'input_drop_out':input_drop_out_box,
                'drop_out':drop_out_box,
                'batch_normalize':batch_normalize_box,
                'test_index':test_index_list ,
                'tr_accuracy':tr_acc_list, 
                'tr_sensitivity':tr_sensitivity_list, 
                'tr_specificity':tr_specificity_list, 
                'ts_accuracy': ts_acc_list,
                'ts_sensitivity':ts_sensitivity_list, 
                'ts_specificity':ts_specificity_list, 
                "tr_auc":tr_auc_list, 
                "ts_auc":ts_auc_list, 
                "total_auc":tot_auc_list}, 
          columns =['index','model_types','model_comb', 'test_index', 'rate', 'count', 'layers','batch_size','input_drop_out','drop_out','batch_normalize', 'tr_accuracy', 'tr_sensitivity', 'tr_specificity', 'ts_accuracy', 'ts_sensitivity', 'ts_specificity', "tr_auc", "ts_auc", "total_auc"])
df_1.to_csv(save_result_path+"Ensembles_"+str(ts_i)+".csv", index=False)

