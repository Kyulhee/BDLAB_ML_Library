#!/usr/bin/env python
# coding: utf-8

# # 0. Importing packages & Defining methods

# ## 1) Importing packages

# In[6]:


from keras.layers import Input, Dense    #using set model component
from keras.models import Model    #using set model 
from keras.utils import plot_model    #show model structure
from keras import layers as Layer
import keras 
from lib import dataProcess as dp
import pandas as pd
from pandas import DataFrame as df
from keras.callbacks import EarlyStopping
import pydot
# prediction


# ## 2) Defining methods

# In[7]:


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


# # 1. Reading data & Preprocessing

# ## 1) Path selection

# In[28]:

count=0

methods = ["old", "new"]
genesets = ["gene_set1", "gene_set2"]
limma_options = ["before_limma", "limma"]

for method in methods:
    for geneset in genesets:
        for limma_option in limma_options:
            count=count+1
            #limma_option = limma_options[int(input("choose the limma_option (1 or 2):\n[1] before_limma\n[2] limma\n"))-1]
            #print("you select "+limma_option)
            #method = methods[int(input("choose the method (1 or 2):\n[1] old\n[2] new\n"))-1]
            #print("you select "+method)
            #geneset = genesets[int(input("choose the geneset (1 or 2):\n[1] gene_set1\n[2] gene_set2\n"))-1]
            #print("you select "+geneset)

            path = "../../Data/"+method+"_set/"
            save_model_path = "../models/"+method+"_set/"
            save_result_path = "../results/"+method+"_set/"

            print("\n###################################################################")
            #print("path: "+path)
            #print("save_model_path: "+save_model_path)
            #print("save_result_path: "+save_result_path)
            print(str(count)+" th train & test: "+method+", "+geneset+", "+limma_option)


            # ## 2) Read data files & Dividing into train / test

            # In[29]:


            train_data=pd.read_csv(path+limma_option+"_TCGA_inter_nonmucinous_"+method+"_"+geneset+"_data.csv")
            test_data=pd.read_csv(path+limma_option+"_TCGA_inter_mucinous_"+method+"_"+geneset+"_data.csv")
            cli_data=pd.read_csv(path+limma_option+"_TCGA_inter_clin_"+method+"_"+geneset+"_data.csv")

            #train_data=train_data.sample(frac=1)
            train_x, train_y = dp.divide_xy_test(train_data, 'result',  1, -1)
            test_x, test_y=dp.divide_xy_test(test_data, 'result',  1, -1)
            cli_x = cli_data.iloc[:, 1:].as_matrix()
            train_y1 = dp.one_hot_encoder_train(train_y)
            test_y1 = dp.one_hot_encoder_test(train_y,test_y)


            # In[30]:


            print("train data: "+str(train_x.shape))
            print("test data: "+str(test_x.shape))
            print("clin data: "+str(cli_x.shape))


            # In[31]:


            if train_x.shape[1] == test_x.shape[1] and test_x.shape[1] == cli_x.shape[1]:
                input_dim = train_x.shape[1]
                print("input dim is "+str(input_dim))
            else:
                print("########################### dimensions are not equal. please check your dataset")
                break

            # In[32]:


            #geneset_merged=keras.layers.Input(shape=(input_dim,))


            # # 2. Establishing & training/testing the DNN model

            # ## 1) Establishing DNN model

            # In[33]:


            nodes_list = [[500,300,100]]
                    
            input_m = keras.layers.Input(shape=(input_dim,))

            h1_m3 = keras.layers.Dense(500,activation="relu")(input_m)
            h1_d = keras.layers.Dropout(0.3,)(h1_m3)
            h2_m3 = keras.layers.Dense(300,activation="relu")(h1_d)
            h2_d = keras.layers.Dropout(0.3,)(h2_m3)
            h3_m3 = keras.layers.Dense(100,activation="relu")(h2_d)
            h3_d = keras.layers.Dropout(0.3,)(h3_m3)
                            
            early_stopping = EarlyStopping(patience = 10)


            # In[34]:


            predictions = Dense(6, activation='softmax', name='predictions')(h3_d)
            model1 = Model(inputs = input_m, output = predictions)
            adam = keras.optimizers.Adam()
            model1.compile(optimizer=adam, loss ='categorical_crossentropy', metrics=['accuracy'])
            model1.summary()


            # ## 2) Training & Testing the DNN model

            # In[35]:


            #plot_model(model, to_file='model.png')
            model1.fit(x=train_x, y=train_y1,callbacks=[early_stopping], epochs = 50,batch_size=25, verbose=0)
            print("train acc: "+str(model1.evaluate(train_x,train_y1, verbose=0)[1])+", test acc: "+str(model1.evaluate(test_x,test_y1, verbose=0)[1]))
            
            # In[36]:


            model1.save(save_model_path+limma_option+"_"+method+"_"+geneset+"_model.h5")



            #model1 = keras.models.load_model(save_model_path+limma_option+"_"+method+"_"+geneset+"_model.h5")
            # ## 3) Predict train & test & clinical dataset

            # In[37]:


            train_h=model1.predict(train_x, batch_size=None, verbose=0, steps=None)
            test_h=model1.predict(test_x, batch_size=None, verbose=0, steps=None)
            cli_h=model1.predict(cli_x, batch_size=None, verbose=0, steps=None)


            # In[38]:


            train_p=[]
            for i in range(len(train_h)):
                train=train_h[i].tolist()
                train_p.append(train.index(max(train)))


            # In[39]:


            test_p=[]
            for i in range(len(test_h)):
                test=test_h[i].tolist()
                test_p.append(test.index(max(test)))


            # In[40]:


            cli_p=[]
            for i in range(len(cli_h)):
                cli=cli_h[i].tolist()
                cli_p.append(cli.index(max(cli)))


            # ## 4) Making result table

            # In[41]:


            df_train_h=df(train_h)
            df_train_h.columns=['CESC','COAD','PAAD','STAD','UCEC','UCS']

            df_train_p=df(train_p)
            df_train_p.columns=['prediction']

            df_train_y=df(train_y)
            df_train_y.columns=['y']

            pd.concat([train_data['sample'],df_train_h,df_train_p,df_train_y],axis=1).to_csv(save_result_path+limma_option+"_nonmucinous_"+method+"_"+geneset+"_train_result.csv",index=False)


            # In[42]:


            df_test_h=df(test_h)
            df_test_h.columns=['CESC','COAD','PAAD','STAD','UCEC','UCS']

            df_test_p=df(test_p)
            df_test_p.columns=['prediction']

            df_test_y=df(test_y)
            df_test_y.columns=['y']

            pd.concat([test_data['sample'],df_test_h,df_test_p,df_test_y],axis=1).to_csv(save_result_path+limma_option+"_mucinous_"+method+"_"+geneset+"_test_result.csv",index=False)


            # In[44]:


            df_cli_h=df(cli_h)
            df_cli_h.columns=['CESC','COAD','PAAD','STAD','UCEC','UCS']

            df_cli_p=df(cli_p)
            df_cli_p.columns=['prediction']

            pd.concat([cli_data['sample'],df_cli_h,df_cli_p],axis=1).to_csv(save_result_path+limma_option+"_clin_"+method+"_"+geneset+"_clin_result.csv",index=False)


            # In[ ]:




