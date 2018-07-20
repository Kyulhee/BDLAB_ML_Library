import pandas as pd
import tensorflow as tf
import numpy as np 
from lib import dataProcess as dp
from lib import deepLearning as dl


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
    sensitivity=result['cancer-correct']/(result['cancer-correct']+result['cancer-wrong'])
    specificity=result['normal-correct']/(result['normal-correct']+result['normal-wrong'])
    print("Sensitivity :", sensitivity)
    print("Specificity :", specificity)

import sys, getopt
def get_options(argv):
    inputfiles = ''
    outputfile = ''
    inputmodels = ''
    nodes = ''
    try:
        opts, args = getopt.getopt(argv,"hi:m:n:o:",["ifiles=", "imodels=","nodes=","ofile="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('help')
            sys.exit()
        elif opt in ("-i", "--ifiles"):
            inputfiles = arg
        elif opt in ("-n", "--nodes"):
            nodes = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-m", "--imodels"):
            inputmodels = arg
        
    print('Input files are ', inputfiles)
    print('Input models are ', inputmodels) 
    print('nodes are ', nodes)
    return inputfiles, inputmodels, nodes

if __name__ == "__main__":
    inputfiles, inputmodels, nodes = get_options(sys.argv[1:])
    inputfiles_f = open(inputfiles, 'r')
    inputmodels_f = open(inputmodels, 'r')
    inputfiles = inputfiles_f.readlines()
    inputmodels = inputmodels_f.readlines()
    inputfiles_f.close()
    inputmodels_f.close()
    ###Setting filelist and modellist
    for i in range(len(inputfiles)):
         inputfile = inputfiles[i]
         inputmodel = inputmodels[i]
         if inputfile[-1] == '\n' or inputfile[-1] == ' ':
            inputfiles[i] = inputfile[:-1]
         if inputmodel[-1] == '\n' or inputmodel[-1] == ' ':
            inputmodels[i] = inputmodel[:-1]
    
    ###Setting hypothesis lists###
    train_prob = pd.DataFrame()
    val_prob = pd.DataFrame()
    test_prob = pd.DataFrame()    
    nodes_l = []
    ###read files and models to get hypothesis
    nodes_f = open(nodes, 'r')
    nodes_l = nodes_f.readlines()
    print(nodes_l)
    nodes_f.close()

    for i in range(len(inputfiles)):
        nodes = nodes_l[i]
        if nodes[-1] == '\n' : 
           nodes = nodes[:-1]
        
        nodes = nodes.split(' ')
        for j in range(len(nodes)):
            nodes[j] = int(nodes[j])
        print(nodes)
        raw_data = pd.read_csv(inputfiles[i])
        fivefold = dp.fivefold(raw_data, 'index')
        xdata_five, ydata_five = dp.divide_xy_train(fivefold, 'result', True, 1, -3)
        train_x, test_x = dp.train_test(xdata_five, 0)
        train_y, test_y = dp.train_test(ydata_five, 0)
        train_y = dp.one_hot_encoder(train_y)
        test_y = dp.one_hot_encoder(test_y)
        val_x, test_x = dp.test_validation(test_x)
        val_y, test_y = dp.test_validation(test_y)
        batch_size = 100
        learning_rate = 0.001

        X, Y, weights, bias, hidden_layers, logits, hypothesis, cost, train, predicted, correct_prediction, accuracy, keep_prob = dl.set_model(
                train_x, train_y, nodes, learning_rate)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, inputmodels[i])
            train_h, train_a = sess.run([hypothesis, accuracy ], feed_dict={X: train_x, Y: train_y, keep_prob: 1.0})
            val_h, val_a = sess.run([hypothesis, accuracy ], feed_dict={X: val_x, Y: val_y, keep_prob: 1.0})
            test_h, test_a = sess.run([hypothesis, accuracy ], feed_dict={X: test_x, Y: test_y, keep_prob: 1.0})
            
            if i == 0 :
                train_prob = pd.DataFrame(train_h, columns=['normal'+str(i), 'cancer'])
                train_prob = train_prob.loc[:,'cancer']
                test_prob = pd.DataFrame(test_h, columns=['normal'+str(i), 'cancer'])
                test_prob = test_prob.loc[:, 'cancer']
                val_prob = pd.DataFrame(val_h, columns=['normal'+str(i), 'cancer'])
                val_prob = val_prob.loc[:, 'cancer']
            else :
                train_h = pd.DataFrame(train_h, columns=['normal'+str(i), 'cancer'])
                train_prob = pd.concat([train_prob, train_h.loc[:,'cancer']], axis=1)
                test_h = pd.DataFrame(test_h, columns=['normal'+str(i), 'cancer'])
                test_prob = pd.concat([test_prob, test_h.loc[:, 'cancer']], axis=1)
                val_h = pd.DataFrame(val_h, columns=['normal'+str(i), 'cancer'])
                val_prob = pd.concat([val_prob, val_h.loc[:, 'cancer']], axis=1)

    train_h_list = train_prob.as_matrix()
    test_h_list = test_prob.as_matrix()
    val_h_list = val_prob.as_matrix()
    print(len(train_h_list[0]))                
    batch_size = 100
    learning_rate = 0.001
    nodes = [20,20,20, 30,30]
    save_path_dir = './'
    X, Y, weights, bias, hidden_layers, logits, hypothesis, cost, train, predicted, correct_prediction, accuracy, keep_prob = dl.set_model( train_h_list, train_y, nodes, learning_rate)

    best_train_acc = 0
    best_val_acc = 0
    best_test_acc = 0 
    best_cost = float("inf")
    save_path_dir ='./'
    count = 0
    model_num = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Initialize TensorFlow variables
        sess.run(tf.global_variables_initializer())
        stop_switch = True
        step = 0

        while stop_switch:
            total_num = int(len(train_h_list) / batch_size)
            for k in range(total_num):
                batch_x = train_h_list[k * batch_size:(k + 1) * batch_size]
                batch_y = train_y[k * batch_size:(k + 1) * batch_size]
                sess.run(train, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})


            train_h, train_c, train_p, train_a = sess.run( [hypothesis, cost, predicted, accuracy], feed_dict={X: train_h_list, Y: train_y, keep_prob: 1})
            val_h, val_c, val_p, val_a = sess.run([hypothesis, cost, predicted, accuracy], feed_dict={X: val_h_list, Y: val_y, keep_prob: 1})
            if step % 20 == 0 :
                print("train acc : ", train_a, "validation acc : ", val_a, "train_cost", train_c)
            step += 1

            if best_cost > val_c :
                best_train_acc = train_a
                best_val_acc = val_a
                best_cost = val_c
                count = 0
                save_path = saver.save(sess, save_path_dir + 'modelensemble.ckpt')

            elif count > 10 :
                stop_switch = False
                print("Learning Finished!! \n")
            else:
                count += 1

        print("Training Accuracy : ", best_train_acc,  "Validation Accuracy : ", best_val_acc)

        saver.restore(sess, save_path)

        test_h, test_p, test_a = sess.run([hypothesis, predicted, accuracy],
                                          feed_dict={X: test_h_list, Y: test_y, keep_prob: 1.0})
        print("\nTest Accuracy: ", test_a)
        best_test_acc = test_a


        check_correct(train_p, np.argmax(train_y, axis=1))
        check_correct(val_p, np.argmax(val_y, axis=1))
        check_correct(test_p , np.argmax(test_y, axis=1))
