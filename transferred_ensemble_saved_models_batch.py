import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Input
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


# In[3]:


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


# In[4]:


# calculate all of model performance 
# - predictions(probability) / labeled predictions(0/1) / Loss / Accuracy / Sensitivity / Specificity / AUC values of Train / Test dataset.
# using trained models, or you can put predictions(probability) passively(in this case, Loss & Accuracy do not provided.)
def model_performance(information=False, Input_Prediction_Passively=False, using_model=None, tr_predictions=None, ts_predictions=None, tr_x_val=None, tr_y_val=None, ts_x_val=None, ts_y_val=None, output_list=None):
    
    if information == True:            
        print("options model_performance:\n1) using_model: keras models that you want to check performance. \"Input_Prediction_Passive\" option for input prediction list instead using models.\"* CAUTION: Essential variable.\n2) tr_x_val & ts_x_val: input samples of train/test samples.\n3) tr_y_val & ts_y_val: results of train/test samples.\n4) output_list: return values that you want to recieve.\n CAUTION: Essential variable.\n\t tr_loss, tr_accuracy, tr_sensitivity, tr_specificity, tr_predictions, labeled_tr_predictions, tr_predictions_flat, roc_auc_tr,\nts_loss, ts_accuracy, ts_sensitivity, ts_specificity, ts_predictions, labeled_ts_predictions, ts_predictions_flat, roc_auc_ts,\nroc_auc_total\n\n* CAUTION: if 'None' value is returned, please check your input tr inputs(None value for tr outputs) or ts inputs(None value for ts outputs).") 
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

types = ["inter_by_names_Annotation3000_400", "inter_by_names_CV_400", 
         "inter_by_names_Var_400", "inter_by_names_new_Diff_400",
         "inter_by_names_Clin", 
         "inter_by_names_SNV" 
         ]

model_path = "G:/내 드라이브/Class/6과 7 사이(hell)/Lab/TCGA 난소암/Best_Models/18.09.15/"
path = "C:/test/TC_intersect_subsamples_by_names/"

# change 'types' and 'load_model' part for using another models.
m_1_name = "Annot_3000_400_0.h5"
m_2_name = "CV_400_0.h5"
m_3_name = "Var_400_0.h5"
m_4_name = "new_Diff_400_0.h5"
m_5_name = "Clin_2.h5"
m_6_name = "SNV_1.h5"
select_types = [types[0],
                types[1],
                types[2],
                types[3],
                types[4],
                types[5]]


# ## Import Data

file_1 = path+select_types[0]+".csv"
file_2 = path+select_types[1]+".csv"
file_3 = path+select_types[2]+".csv"
file_4 = path+select_types[3]+".csv"
file_5 = path+select_types[4]+".csv"
file_6 = path+select_types[5]+".csv"

idx_col = 0

data_1 = pd.read_csv(file_1,index_col=idx_col)
data_2 = pd.read_csv(file_2,index_col=idx_col)
data_3 = pd.read_csv(file_3,index_col=idx_col)
data_4 = pd.read_csv(file_4,index_col=idx_col)
data_5 = pd.read_csv(file_5,index_col=idx_col)
data_6 = pd.read_csv(file_6,index_col=idx_col)

sample_1,features_1 = data_1.shape
sample_2,features_2 = data_2.shape
sample_3,features_3 = data_3.shape
sample_4,features_4 = data_4.shape
sample_5,features_5 = data_5.shape
sample_6,features_6 = data_6.shape

# Data frame include index & Platinum_Status column, substract 2 to calculate real number of features 
[features_1, features_2, features_3, features_4, features_5, features_6] = [features_1-2, features_2-2, features_3-2, features_4-2, features_5-2, features_6-2]

# Split Train Test Data

train_data_1, test_data_1, y_val_1, x_val_1, test_y_val_1, test_x_val_1 = data_split(raw_data = data_1, index_col = -1, test_index = 1)
train_data_2, test_data_2, y_val_2, x_val_2, test_y_val_2, test_x_val_2 = data_split(raw_data = data_2, index_col = -1, test_index = 1)
train_data_3, test_data_3, y_val_3, x_val_3, test_y_val_3, test_x_val_3 = data_split(raw_data = data_3, index_col = -1, test_index = 1)
train_data_4, test_data_4, y_val_4, x_val_4, test_y_val_4, test_x_val_4 = data_split(raw_data = data_4, index_col = -1, test_index = 1)
train_data_5, test_data_5, y_val_5, x_val_5, test_y_val_5, test_x_val_5 = data_split(raw_data = data_5, index_col = -1, test_index = 1)
train_data_6, test_data_6, y_val_6, x_val_6, test_y_val_6, test_x_val_6 = data_split(raw_data = data_6, index_col = -1, test_index = 1)

model_1_l = load_model(model_path+m_1_name)
m_1_l_tr_loss,m_1_l_tr_accuracy=model_1_l.evaluate(x_val_1,y_val_1)
m_1_l_loss,m_1_l_accuracy= model_1_l.evaluate(test_x_val_1,test_y_val_1)
model_1_l_new = Model(inputs = model_1_l.input, outputs=model_1_l.get_layer(model_1_l.layers[-2].name).output)

model_2_l = load_model(model_path+m_2_name)
m_2_l_tr_loss,m_2_l_tr_accuracy=model_2_l.evaluate(x_val_2,y_val_2)
m_2_l_loss,m_2_l_accuracy= model_2_l.evaluate(test_x_val_2,test_y_val_2)
model_2_l_new = Model(inputs = model_2_l.input, outputs=model_2_l.get_layer(model_2_l.layers[-2].name).output)

model_3_l = load_model(model_path+m_3_name)
m_3_l_tr_loss,m_3_l_tr_accuracy=model_3_l.evaluate(x_val_3,y_val_3)
m_3_l_loss,m_3_l_accuracy= model_3_l.evaluate(test_x_val_3,test_y_val_3)
model_3_l_new = Model(inputs = model_3_l.input, outputs=model_3_l.get_layer(model_3_l.layers[-2].name).output)

model_4_l = load_model(model_path+m_4_name)
m_4_l_tr_loss,m_4_l_tr_accuracy=model_4_l.evaluate(x_val_4,y_val_4)
m_4_l_loss,m_4_l_accuracy= model_4_l.evaluate(test_x_val_4,test_y_val_4)
model_4_l_new = Model(inputs = model_4_l.input, outputs=model_4_l.get_layer(model_4_l.layers[-2].name).output)

model_5_l = load_model(model_path+m_5_name)
m_5_l_tr_loss,m_5_l_tr_accuracy=model_5_l.evaluate(x_val_5,y_val_5)
m_5_l_loss,m_5_l_accuracy= model_5_l.evaluate(test_x_val_5,test_y_val_5)
model_5_l_new = Model(inputs = model_5_l.input, outputs=model_5_l.get_layer(model_5_l.layers[-2].name).output)

model_6_l = load_model(model_path+m_6_name)
m_6_l_tr_loss,m_6_l_tr_accuracy=model_6_l.evaluate(x_val_6,y_val_6)
m_6_l_loss,m_6_l_accuracy= model_6_l.evaluate(test_x_val_6,test_y_val_6)
model_6_l_new = Model(inputs = model_6_l.input, outputs=model_6_l.get_layer(model_6_l.layers[-2].name).output)

m_1_predictions = model_1_l.predict(x_val_1)
m_2_predictions = model_2_l.predict(x_val_2)
m_3_predictions = model_3_l.predict(x_val_3)
m_4_predictions = model_4_l.predict(x_val_4)
m_5_predictions = model_5_l.predict(x_val_5)
m_6_predictions = model_6_l.predict(x_val_6)

m_1_test_predictions = model_1_l.predict(test_x_val_1)
m_2_test_predictions = model_2_l.predict(test_x_val_2)
m_3_test_predictions = model_3_l.predict(test_x_val_3)
m_4_test_predictions = model_4_l.predict(test_x_val_4)
m_5_test_predictions = model_5_l.predict(test_x_val_5)
m_6_test_predictions = model_6_l.predict(test_x_val_6)


# select models for ensemble among loaded models.
# CAUTION: Duplication(ex: select = [1, 1, 1, 3, 5]) is allowed, but it is same models, and have same predictions. They have same opinions.

#select = [1, 4, 5, 6]

em_type_box = []
index_box = []
model_names_box = []
tr_accuracy_box = []
ts_accuracy_box = []
tr_sensitivity_box = []
ts_sensitivity_box = []
tr_specificity_box = []
ts_specificity_box = []
tr_auc_box = []
ts_auc_box = []
total_auc_box = []
t_layer_box = []
learning_rate_box = []

'''
selects = [[1, 2, 3, 4], [1, 2, 3, 5]]
t_layers = [[50, 50, 50], [100, 50, 20]]
learning_rates = [0.01, 0.005, 0.001]
'''
selects = [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6], [1, 2, 4, 5], [1, 2, 4, 6], [1, 2, 5, 6], [1, 3, 4, 5], [1, 3, 4, 6], [1, 3, 5, 6], [1, 4, 5, 6],
           [2, 3, 4, 5], [2, 3, 4, 6], [2, 3, 5, 6], [2, 4, 5, 6],
           [3, 4, 5, 6]]
t_layers = [[50, 50, 50], [100, 50, 20], [100, 100, 100], [200, 100, 50], [100, 200, 50] , [100, 100, 100, 100]]
learning_rates = [0.01, 0.005, 0.001]

num = 0

for select in selects:
    print(select)
    model_names = []
    for t in range(len(select)):
        n = select_types[select[t]-1].split('_')
        print(n)
        a=n[3]
        for k in range(4, len(n)):
            a = a+"_"+n[k]
        model_names.append(a)
    print(model_names)

    for t_layer in t_layers:
        for learning_rate in learning_rates:
                                
            # ## 1) DNN-Combiner Ensmeble

            # ### Building original ensemble model

            print("################################## DNN em ##################################")
            print("select: "+str(select))
            for select_type_i in select:
                print(select_types[select_type_i-1])
            print("#############################################################################################")

            # 1) parameter setting
            adam = optimizers.Adam(lr=learning_rate)
            input_drop_out_em = 0.5
            drop_out_em = 0.5
            layers = [5]
            em_tr_loss_best = 100 # for saving best loss value 
            best_em_model=[] #for saving best model
            count=0 # for early stopping


            # 2) model build
            m_tr_predictions = [m_1_predictions, m_2_predictions, m_3_predictions, m_4_predictions, m_5_predictions, m_6_predictions]
            m_tr_predictions_select = []

            for i in range(len(select)):
                m_tr_predictions_select.append(m_tr_predictions[select[i]-1])
                #print(m_tr_predictions[select[i]-1].shape)
                
            em_x_val = np.concatenate(m_tr_predictions_select, axis=1)

            input_em = Input(shape=(len(select),))
            em_m_dp = Dropout(input_drop_out_em)(input_em)
            for i in layers:
                em_m = Dense(i,activation='relu')(em_m_dp)
                em_m_dp = Dropout(drop_out_em)(em_m)
            em_m_final = em_m_dp
            output_em = Dense(1, activation="sigmoid")(em_m_final)
            em_model = Model(inputs=input_em,outputs=output_em)
            em_model.compile(optimizer=adam, 
                            loss='binary_crossentropy',
                            metrics=['accuracy'])


            # 3) Training: if no increase of tr_loss three times, stop training.
            max_epochs = 0
            while max_epochs < 20:
                em_model.fit(em_x_val, y_val_1, batch_size=10, epochs=1, verbose = 0)
                em_tr_loss=em_model.evaluate(em_x_val,y_val_1)[0]
                max_epochs = max_epochs +1
                if em_tr_loss < em_tr_loss_best: # new best model. count reset.
                    em_tr_loss_best = em_tr_loss
                    count=0
                    best_em_model = em_model
                if count>3: # no increase three time. stop.
                    em_model = best_em_model
                    break
                else: count=count+1

            # 4) save model
            em_model.save("../models/Ovary/m_em_"+str(num)+".h5")

            # ### Evaluating _DNN Combiner_ ensemble model

            m_test_predictions = [m_1_test_predictions, m_2_test_predictions, m_3_test_predictions, m_4_test_predictions, m_5_test_predictions, m_6_test_predictions]
            m_test_predictions_select = []   
            #def prediction_result(pathway, tr_x_val, tr_y_val, ts_x_val, ts_y_val, tr_patients, ts_patients)
            for i in range(len(select)):
                m_test_predictions_select.append(m_test_predictions[select[i]-1])
                
            em_test_x_val = np.concatenate(m_test_predictions_select, axis=1)

            em_output_list = model_performance(
                information = False, using_model=em_model,Input_Prediction_Passively = False, 
                tr_x_val=em_x_val, tr_y_val=y_val_1, ts_x_val=em_test_x_val, ts_y_val=test_y_val_1,
                output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                             "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                             "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                             "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                             "roc_auc_total"])

            em_tr_loss, em_tr_accuracy, em_tr_sensitivity, em_tr_specificity, em_tr_predictions, em_labeled_tr_predictions, em_tr_predictions_flat, em_roc_auc_tr, em_ts_loss, em_ts_accuracy, em_ts_sensitivity, em_ts_specificity, em_ts_predictions,em_labeled_ts_predictions, em_ts_predictions_flat, em_roc_auc_ts, em_roc_auc_total = em_output_list

            # save prediction result.

            tr_df_em = pd.DataFrame(data={"patient":list(train_data_1.index), "hypothesis 1": list(em_tr_predictions_flat), 
                                    "prediction":list(em_labeled_tr_predictions), "Platinum_Status":list(y_val_1)})
            tr_df_em.to_csv("../result/prediction_result_DNN_em_tr_"+str(num)+".csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

            ts_df_em = pd.DataFrame(data={"patient":list(test_data_1.index), "hypothesis 1": list(em_ts_predictions_flat), 
                                    "prediction":list(em_labeled_ts_predictions), "Platinum_Status":list(test_y_val_1)})
            ts_df_em.to_csv("../result/prediction_result_DNN_em_ts_"+str(num)+".csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])
            

            # ## 2) Mean Ensemble

            # ### Evaluating _mean_ ensemble model

            mean_em_tr_predictions=sum(m_tr_predictions_select)/len(select)
            mean_em_ts_predictions=sum(m_test_predictions_select)/len(select)

            mean_em_output_list = model_performance(
                information = False, using_model=None,Input_Prediction_Passively = True, 
                tr_predictions=mean_em_tr_predictions, ts_predictions=mean_em_ts_predictions, 
                tr_x_val=em_x_val, tr_y_val=y_val_1, ts_x_val=em_test_x_val, ts_y_val=test_y_val_1,
                output_list=["tr_sensitivity", "tr_specificity",
                             "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                             "ts_sensitivity", "ts_specificity",
                             "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                             "roc_auc_total"])
            mean_em_tr_sensitivity, mean_em_tr_specificity,  mean_em_labeled_tr_predictions, mean_em_tr_predictions_flat, mean_em_roc_auc_tr, mean_em_ts_sensitivity, mean_em_ts_specificity, mean_em_labeled_ts_predictions, mean_em_ts_predictions_flat, mean_em_roc_auc_ts, mean_em_roc_auc_total = mean_em_output_list

            mean_em_tr_accuracy = sum(mean_em_labeled_tr_predictions==y_val_1.values)/len(y_val_1)
            mean_em_ts_accuracy = sum(mean_em_labeled_ts_predictions==test_y_val_1.values)/len(test_y_val_1)

            # save prediction result.

            tr_df_mean_em = pd.DataFrame(data={"patient":list(train_data_1.index), "hypothesis 1": list(mean_em_tr_predictions_flat), 
                                    "prediction":list(mean_em_labeled_tr_predictions), "Platinum_Status":list(y_val_1)})
            tr_df_mean_em.to_csv("../result/prediction_result_mean_em_tr_"+str(num)+".csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

            ts_df_mean_em = pd.DataFrame(data={"patient":list(test_data_1.index), "hypothesis 1": list(mean_em_ts_predictions_flat), 
                                    "prediction":list(mean_em_labeled_ts_predictions), "Platinum_Status":list(test_y_val_1)})
            ts_df_mean_em.to_csv("../result/prediction_result_mean_em_ts_"+str(num)+".csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])


            # ## 3) Transferred Ensemble Modeling 

            # ### Making new input data for t-ensemble
            results_m_1 = model_1_l_new.predict([x_val_1])
            results_m_2 = model_2_l_new.predict([x_val_2])
            results_m_3 = model_3_l_new.predict([x_val_3])
            results_m_4 = model_4_l_new.predict([x_val_4])
            results_m_5 = model_5_l_new.predict([x_val_5])
            results_m_6 = model_6_l_new.predict([x_val_6])

            results_m_sum = [results_m_1, results_m_2, results_m_3, results_m_4, results_m_5, results_m_6]
            results_m_select = []

            for i in range(len(select)):
                results_m_select.append(results_m_sum[select[i]-1])

            t_em_x_val = np.concatenate(results_m_select, axis=1)
            print(t_em_x_val.shape)


            # ### Modeling t-ensemble  

            # 1) parameter setting
            adam = optimizers.Adam(lr=learning_rate)
            input_drop_out_t_em = 0.5
            drop_out_t_em = 0.5
            layers = t_layer
            t_em_tr_loss_best = 100 # for saving best loss value 
            best_t_em_model=[] #for saving best model
            count=0 # for early stopping


            # 2) model build
            input_t_em = Input(shape=(t_em_x_val.shape[1],))
            t_em_m_dp = Dropout(input_drop_out_t_em)(input_t_em)
            for i in layers:
                t_em_m = Dense(i,activation='relu')(t_em_m_dp)
                t_em_m_dp = Dropout(drop_out_em)(t_em_m)
            t_em_m_final = t_em_m_dp
            output_t_em = Dense(1, activation="sigmoid")(t_em_m_final)
            t_em_model = Model(inputs=input_t_em,outputs=output_t_em)
            t_em_model.compile(optimizer=adam, 
                            loss='binary_crossentropy',
                            metrics=['accuracy'])


            # 3) Training: if no increase of tr_loss three times, stop training.

            #t_em_model.fit(t_em_x_val, y_val_1, batch_size=5, epochs = 100, validation_split = 0.1, callbacks=[early_stopping])
            max_epochs = 0
            while max_epochs < 20:
                t_em_model.fit(t_em_x_val, y_val_1, batch_size=10, epochs=1, verbose = 0)
                t_em_tr_loss=t_em_model.evaluate(t_em_x_val,y_val_1)[0]
                max_epochs = max_epochs+1
                if t_em_tr_loss < t_em_tr_loss_best: # new best model. count reset.
                    t_em_tr_loss_best = t_em_tr_loss
                    count=0
                    best_t_em_model = t_em_model
                if count>3: # no increase three time. stop.
                    t_em_model = best_t_em_model
                    break
                else: count=count+1

            print("transffered ensemble model trained.")
            t_em_model.save("../models/Ovary/t_em_"+str(num)+".h5")


            # ### Evaluating t-ensemble

            test_results_m_1 = model_1_l_new.predict([test_x_val_1])
            test_results_m_2 = model_2_l_new.predict([test_x_val_2])
            test_results_m_3 = model_3_l_new.predict([test_x_val_3])
            test_results_m_4 = model_4_l_new.predict([test_x_val_4])
            test_results_m_5 = model_5_l_new.predict([test_x_val_5])
            test_results_m_6 = model_6_l_new.predict([test_x_val_6])

            test_results_m_sum = [test_results_m_1, test_results_m_2, test_results_m_3, test_results_m_4, test_results_m_5, test_results_m_6]
            test_results_m_select = []

            for i in range(len(select)):
                test_results_m_select.append(test_results_m_sum[select[i]-1])

            t_em_test_x_val = np.concatenate(test_results_m_select, axis=1)

            t_em_output_list = model_performance(
                information = False, using_model=t_em_model,Input_Prediction_Passively = False, 
                tr_x_val=t_em_x_val, tr_y_val=y_val_1, ts_x_val=t_em_test_x_val, ts_y_val=test_y_val_1,
                output_list=["tr_loss", "tr_accuracy", "tr_sensitivity", "tr_specificity", "tr_predictions",
                             "labeled_tr_predictions", "tr_predictions_flat", "roc_auc_tr", 
                             "ts_loss", "ts_accuracy", "ts_sensitivity", "ts_specificity", "ts_predictions",
                             "labeled_ts_predictions", "ts_predictions_flat", "roc_auc_ts", 
                             "roc_auc_total"])

            t_em_tr_loss, t_em_tr_accuracy, t_em_tr_sensitivity, t_em_tr_specificity, t_em_tr_predictions, t_em_labeled_tr_predictions, t_em_tr_predictions_flat, t_em_roc_auc_tr, t_em_ts_loss, t_em_ts_accuracy, t_em_ts_sensitivity, t_em_ts_specificity, t_em_ts_predictions,t_em_labeled_ts_predictions, t_em_ts_predictions_flat, t_em_roc_auc_ts, t_em_roc_auc_total = t_em_output_list

            tr_df_t_em = pd.DataFrame(data={"patient":list(train_data_1.index), "hypothesis 1": list(t_em_tr_predictions_flat), 
                                    "prediction":list(t_em_labeled_tr_predictions), "Platinum_Status":list(y_val_1)})
            tr_df_t_em.to_csv("../result/prediction_result_t_em_tr_"+str(num)+".csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

            ts_df_t_em = pd.DataFrame(data={"patient":list(test_data_1.index), "hypothesis 1": list(t_em_ts_predictions_flat), 
                                    "prediction":list(t_em_labeled_ts_predictions), "Platinum_Status":list(test_y_val_1)})
            ts_df_t_em.to_csv("../result/prediction_result_t_em_ts_"+str(num)+".csv", index=False, header=True, columns = ["patient", "hypothesis 1", "prediction", "Platinum_Status"])

            em_type_box.append("DNN_em")
            index_box.append(num)
            model_names_box.append(model_names)
            tr_accuracy_box.append(em_tr_accuracy)
            ts_accuracy_box.append(em_ts_accuracy)
            tr_sensitivity_box.append(em_tr_sensitivity)
            ts_sensitivity_box.append(em_ts_sensitivity)
            tr_specificity_box.append(em_tr_specificity)
            ts_specificity_box.append(em_ts_specificity)
            tr_auc_box.append(em_roc_auc_tr)
            ts_auc_box.append(em_roc_auc_ts)
            total_auc_box.append(em_roc_auc_total)
            t_layer_box.append("NULL")
            learning_rate_box.append(learning_rate)

            em_type_box.append("mean_em")
            index_box.append(num)
            model_names_box.append(model_names)
            tr_accuracy_box.append(mean_em_tr_accuracy)
            ts_accuracy_box.append(mean_em_ts_accuracy)
            tr_sensitivity_box.append(mean_em_tr_sensitivity)
            ts_sensitivity_box.append(mean_em_ts_sensitivity)
            tr_specificity_box.append(mean_em_tr_specificity)
            ts_specificity_box.append(mean_em_ts_specificity)
            tr_auc_box.append(mean_em_roc_auc_tr)
            ts_auc_box.append(mean_em_roc_auc_ts)
            total_auc_box.append(mean_em_roc_auc_total)
            t_layer_box.append("NULL")
            learning_rate_box.append("NULL")

            em_type_box.append("t_em")
            index_box.append(num)
            model_names_box.append(model_names)
            tr_accuracy_box.append(t_em_tr_accuracy)
            ts_accuracy_box.append(t_em_ts_accuracy)
            tr_sensitivity_box.append(t_em_tr_sensitivity)
            ts_sensitivity_box.append(t_em_ts_sensitivity)
            tr_specificity_box.append(t_em_tr_specificity)
            ts_specificity_box.append(t_em_ts_specificity)
            tr_auc_box.append(t_em_roc_auc_tr)
            ts_auc_box.append(t_em_roc_auc_ts)
            total_auc_box.append(t_em_roc_auc_total)
            t_layer_box.append(t_layer)
            learning_rate_box.append(learning_rate)

            num = num+1
            
df_sum = pd.DataFrame(data={"index":index_box, "model_names": model_names_box, "em_type":em_type_box ,"tr_accuracy": tr_accuracy_box ,"ts_accuracy": ts_accuracy_box,"tr_sensitivity": tr_sensitivity_box,"ts_sensitivity": ts_sensitivity_box,"tr_specificity": tr_specificity_box,"ts_specificity": ts_specificity_box,"tr_auc": tr_auc_box,"ts_auc": ts_auc_box,"total_auc": total_auc_box,"t_layer": t_layer_box,"learning_rate": learning_rate_box})

df_sum.to_csv("../result/Ensemble_accuracy_result.csv", index=False, header=True, columns = ["model_names", "index","em_type","t_layer","learning_rate","tr_accuracy","ts_accuracy","tr_sensitivity","ts_sensitivity","tr_specificity","ts_specificity","tr_auc","ts_auc","total_auc"])          
