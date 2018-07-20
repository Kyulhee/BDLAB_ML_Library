import pandas as pd
import tensorflow as tf
import numpy as np 
from lib import dataProcess as dp
from lib import deepLearning as dl

import sys, getopt
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
    print('Nodes ', nodes)
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
    test_prob = pd.DataFrame()
    nodes_l = []
    ###read files and models to get hypothesis
    nodes_f = open(nodes, 'r')
    nodes_l = nodes_f.readlines()
    print(nodes_l)
    nodes_f.close()
    ids = None
    cancer_code = None
    result = None    
    ###read files and models to get hypothesis
    for i in range(len(inputfiles)):
        nodes = nodes_l[i]
        if nodes[-1] == '\n' :
           nodes = nodes[:-1]

        nodes = nodes.split(' ')
        for j in range(len(nodes)):
            nodes[j] = int(nodes[j])
        raw_data = pd.read_csv(inputfiles[i])
        ids = raw_data['patient']
        cancer_code = raw_data['cancer_code']
        result = raw_data['result']
        test_x, test_y = dp.divide_xy_test(raw_data, 'result', 1, -3)
        test_y = dp.one_hot_encoder(test_y)
        batch_size = 100
        learning_rate = 0.001

        X, Y, weights, bias, hidden_layers, logits, hypothesis, cost, train, predicted, correct_prediction, accuracy, keep_prob = dl.set_model(
                test_x,test_y, nodes, learning_rate)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, inputmodels[i])
            test_h, test_a = sess.run([hypothesis, accuracy ], feed_dict={X: test_x, Y: test_y, keep_prob: 1.0})
            
            if i == 0 :
                test_prob = pd.DataFrame(test_h, columns=['normal'+str(i), 'cancer'])
                test_prob = test_prob.loc[:, 'cancer']
            else :
                test_h = pd.DataFrame(test_h, columns=['normal'+str(i), 'cancer'])
                test_prob = pd.concat([test_prob, test_h.loc[:, 'cancer']], axis=1)

    test_h_list = test_prob.as_matrix()


    print(len(test_h_list[0]))
                
    batch_size = 100
    learning_rate = 0.001
    nodes = [20,20,20,30,30]
    save_path_dir = './'
    X, Y, weights, bias, hidden_layers, logits, hypothesis, cost, train, predicted, correct_prediction, accuracy, keep_prob = dl.set_model( test_h_list, test_y, nodes, learning_rate)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize TensorFlow variables
        saver.restore(sess, 'modelensemble.ckpt')
        test_p,  test_h, test_a = sess.run([predicted, hypothesis, accuracy ], feed_dict={X: test_h_list, Y: test_y, keep_prob: 1.0})
        print(test_a)
        check_correct(test_p, np.argmax(test_y, axis=1))
    ids = ids.to_frame()
    result = result.to_frame()
    cancer_code = cancer_code.to_frame()
    test_p = pd.DataFrame(test_p, columns =['prediction'])
    output = pd.concat([ids, cancer_code, test_p, result], axis = 1)
    output.to_csv('test.csv' , index=False )
