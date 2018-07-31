import pandas as pd
import tensorflow as tf
import lib.dataProcess as dp
import lib.deepLearning as dl
from pandas import DataFrame as df

file_types = [

    #"Var_100", "CV_100",
    "new_Diff_100",
    #"Annotation3000_100",
    #"Var_200", "CV_200",
    "new_Diff_200",
    #"Annotation3000_200",
    #"Var_400", "CV_400",
    "new_Diff_400",
    #"Annotation3000_400",
    #"Var_1000", "CV_1000",
    "new_Diff_1000",
    #"Annotation3000_1000",
    #"Clin_ch"
	#"Diff_100", "Diff_200", "Diff_400", 
	#"Annotation40"
	#"Clin"
	]

num=0


for file_type in file_types:
    file_name = "inter_OV_"+file_type+".csv"
    print("file type: "+file_type)

    raw_data = pd.read_csv('../subsamples/'+file_name)

    shuffled_data = raw_data
    shuffled_data['index'] = dp.shuffle_index(shuffled_data)

    # make index as rep of 1:5
    fivefold = dp.n_fold(raw_data, 'index', 5)

    # devide train & test
    xdata_five, ydata_five = dp.divide_xy_train(fivefold, 'Platinum_Status', True, 1, -2)
    train_x, test_x = dp.train_test(xdata_five, 0)
    train_y, test_y = dp.train_test(ydata_five, 0)
    train_y = dp.one_hot_encoder(train_y)
    test_y = dp.one_hot_encoder(test_y)
    val_x, test_x = dp.test_validation(test_x)
    val_y, test_y = dp.test_validation(test_y)

    #set hyperparameters - node, learning rate, batch size
    '''
    nodes = [[200,200,200], [200,300,200], [300, 300, 300], [100, 100, 100, 100]]
    learning_rates = [0.01, 0.005, 0.001, 0.0005]
    batch_sizes = [10, 50, 75, 100]
    '''
    #for larger data
    #'''
    nodes = [[400,400,400], [150,200,200,150], [150, 150, 150, 150], [150, 200, 300, 400]]
    learning_rates = [0.01, 0.005, 0.001, 0.0005]
    batch_sizes = [10, 50, 75, 100]
    #'''
    #for fast experiment
    '''
    nodes = [[100,150,200]]
    learning_rates = [0.001]
    batch_sizes = [20]
    '''

    nodes_box = []
    learning_rate_box = []
    batch_size_box = []
    tr_accuracy_box = []
    ts_accuracy_box = []
    val_accuracy_box = []
    tr_sensitivity_box = []
    ts_sensitivity_box = []
    val_sensitivity_box = []
    tr_specificity_box = []
    ts_specificity_box = []
    val_specificity_box = []
    index = []
    
    tr_TPs = []
    tr_TNs = []
    tr_FPs = []
    tr_FNs = []
    ts_TPs = []
    ts_TNs = []
    ts_FPs = []
    ts_FNs = []
    val_TPs = []
    val_TNs = []
    val_FPs = []
    val_FNs = []

    for node in nodes:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:

                #make place holders. These are not real variables, just spaces for variable.
                X, Y, layers, logits, phase, hypothesis, cost, train, predicted, correct_prediction, accuracy, keep_prob = dl.set_model_dropout(train_x, train_y, node , learning_rate)

                best_train_acc = 0
                best_val_acc = 0
                best_test_acc = 0 
                best_cost = float("inf")
                save_path_dir ='../models/'
                count = 0
                model_num = 0

                saver = tf.train.Saver()
                with tf.Session() as sess:
                    # Initialize TensorFlow variables
                    sess.run(tf.global_variables_initializer())
                    stop_switch = True
                    step = 0
                    
                    #if condition 
                    while stop_switch:
                        total_num = int(len(train_x) / batch_size)
                        for k in range(total_num):
                            #cut data as large as batch_size.
                            batch_x = train_x[k * batch_size:(k + 1) * batch_size]
                            batch_y = train_y[k * batch_size:(k + 1) * batch_size]
                            #dropout_ratio
                            sess.run(train, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.7 , phase:True})

                        #feed_dict - place holder is just 'space', feed_dict means supply real data into place holder.
                        train_h, train_c, train_p, train_a = sess.run( [hypothesis, cost, predicted, accuracy], feed_dict={X: train_x, Y: train_y, keep_prob: 1 , phase:False})
                        val_h, val_c, val_p, val_a = sess.run([hypothesis, cost, predicted, accuracy], feed_dict={X: val_x, Y: val_y, keep_prob: 1 , phase:False})
                        if step % 20 == 0 :
                            print("train acc : ", train_a, "validation acc : ", val_a, "train_cost", train_c)
                        step += 1
                        
                        #condition 1: new best condition. 
                        #calculated cost(val_c) is smaller than before's(best_cost), save present condition and initialize count.
                        if best_cost > val_c :
                            best_train_acc = train_a
                            best_train_p = train_p
                            best_val_acc = val_a
                            best_val_p = val_p
                            best_cost = val_c
                            count = 0
                            save_path = saver.save(sess, save_path_dir + 'model'+str(num)+'.ckpt')

                        #condition 2: cannot find best condition
                        #when cost is get worse and worse(10 times), finish learning.
                        elif count > 10 :
                            stop_switch = False
                            print("Learning Finished!! \n")
                        
                        #condition 3: condition is not best, but not yet finish.
                        else:
                            count += 1

                    print("########## Training Accuracy : ", best_train_acc,  "Validation Accuracy : ", best_val_acc)

                    saver.restore(sess, save_path)

                    test_h, test_p, test_a = sess.run([hypothesis, predicted, accuracy],
                                                      feed_dict={X: test_x, Y: test_y, keep_prob:1.0 , phase:False})
                    print("########## Test Accuracy: ", test_a)
                    best_test_acc = test_a

                    model_num += 1
                    train_label, test_label = dp.train_test(ydata_five, 0)
                    val_label, test_label = dp.test_validation_label(test_label)

                with tf.Session() as sess:
                    tr_table_temp = tf.confusion_matrix(train_label, best_train_p, num_classes=2, dtype=tf.int64, name=None, weights=None)
                    val_table_temp = tf.confusion_matrix(val_label, best_val_p, num_classes=2, dtype=tf.int64, name=None, weights=None)
                    ts_table_temp = tf.confusion_matrix(test_label, test_p, num_classes=2, dtype=tf.int64, name=None, weights=None)
                    tr_table, val_table, ts_table = sess.run([tr_table_temp, val_table_temp, ts_table_temp])
                
                # save contingency table
                tr_TN, tr_FN, tr_FP, tr_TP = tr_table[0,0], tr_table[1,0], tr_table[0,1], tr_table[1,1]
                tr_TPs.append(tr_TP)
                tr_TNs.append(tr_TN)
                tr_FPs.append(tr_FP)
                tr_FNs.append(tr_FN)

                ts_TN, ts_FN, ts_FP, ts_TP = ts_table[0,0], ts_table[1,0], ts_table[0,1], ts_table[1,1]
                ts_TPs.append(ts_TP)
                ts_TNs.append(ts_TN)
                ts_FPs.append(ts_FP)
                ts_FNs.append(ts_FN)

                val_TN, val_FN, val_FP, val_TP = val_table[0,0], val_table[1,0], val_table[0,1], val_table[1,1]
                val_TPs.append(val_TP)
                val_TNs.append(val_TN)
                val_FPs.append(val_FP)
                val_FNs.append(val_FN)
     
                    
                data_x, data_y = dp.divide_xy_test(raw_data, 'Platinum_Status', 1,-2)
                ids = raw_data['patient']

                saver = tf.train.Saver()
                with tf.Session() as sess:
                    saver.restore(sess, '../models/model'+str(num)+'.ckpt')
                    test_h, test_p = sess.run([hypothesis, predicted], feed_dict={X: data_x, Y: dp.one_hot_encoder(data_y), keep_prob:1.0 , phase:False})
                print(test_p)

                index.append(num)
                
                # prediction result(probability) file export
                dp.prediction_probability(test_h, test_p, data_y, ids, num)
                num = num+1
                
                # save hyperparameter
                nodes_box.append(node)
                learning_rate_box.append(learning_rate)
                batch_size_box.append(batch_size)

                # save accuracy & sensitivity & specificity
                tr_accuracy_box.append(best_train_acc)
                tr_sensitivity_box.append(tr_TP/(tr_TP+tr_FN))
                tr_specificity_box.append(tr_TN/(tr_TN+tr_FP))
               
                ts_accuracy_box.append(best_test_acc)
                ts_sensitivity_box.append(ts_TP/(ts_TP+ts_FN))
                ts_specificity_box.append(ts_TN/(ts_TN+ts_FP))
                
                val_accuracy_box.append(best_val_acc)
                val_sensitivity_box.append(val_TP/(val_TP+val_FN))
                val_specificity_box.append(val_TN/(val_TN+val_FP))
                

                
    #train y, train 
    df1 = df(data={'index': index, 'nodes': nodes_box, 'learning_rate': learning_rate_box, 'batch_sizes': batch_size_box, 'tr_accuracy':tr_accuracy_box, 'tr_sensitivity':tr_sensitivity_box, 'tr_specificity':tr_specificity_box, 'val_accuracy':val_accuracy_box, 'val_sensitivity':val_sensitivity_box, 'val_specificity':val_specificity_box, 'ts_accuracy': ts_accuracy_box, 'ts_sensitivity':ts_sensitivity_box, 'ts_specificity':ts_specificity_box })
    df1.to_csv("../result/OV_DNN_result_"+file_type+".csv", index=False, header=True, mode='w')
    df2 = df(data={'index': index, 'nodes': nodes_box, 'learning_rate': learning_rate_box, 'batch_sizes': batch_size_box, 'tr_TP': tr_TPs, 'tr_TN': tr_TNs, 'tr_FP': tr_FPs, 'tr_FN': tr_FNs, 'val_TP': val_TPs, 'val_TN': val_TNs, 'val_FP': val_FPs, 'val_FN': val_FNs, 'ts_TP': ts_TPs, 'ts_TN': ts_TNs, 'ts_FP': ts_FPs, 'ts_FN': ts_FNs})
    df2.to_csv("../result/OV_DNN_result_"+file_type+"_raw.csv", index=False, header=True, mode='w')

    


len(test_h)
len(data_y)
len(ids)
len(test_p)
test_p
test_h
