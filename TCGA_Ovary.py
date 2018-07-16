import pandas as pd
import tensorflow as tf
import lib.dataProcess as dp
import lib.deepLearning as dl

file_names = ["OV_Var_200.csv", "OV_Diff_200.csv", "OV_CV_200.csv", "OV_Annotation3000_200.csv", "OV_Annotation40.csv"]
file_name = file_names[1]
file_name

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
nodes = [200,200,200]
learning_rate = 0.001
batch_size = 100

#make place holders. These are not real variables, just spaces for variable.
X, Y, layers, logits, phase, hypothesis, cost, train, predicted, correct_prediction, accuracy, keep_prob = dl.set_model_dropout(train_x, train_y, nodes , learning_rate)

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
    
    #if condition 
    while stop_switch:
        total_num = int(len(train_x) / batch_size)
        for k in range(total_num):
            #cut data as large as batch_size.
            batch_x = train_x[k * batch_size:(k + 1) * batch_size]
            batch_y = train_y[k * batch_size:(k + 1) * batch_size]
            #dropout_ratio
            sess.run(train, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5 , phase:True})

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
            best_val_acc = val_a
            best_cost = val_c
            count = 0
            save_path = saver.save(sess, save_path_dir + 'model'+str(model_num)+'.ckpt')

        #condition 2: cannot find best condition
        #when cost is get worse and worse(10 times), finish learning.
        elif count > 10 :
            stop_switch = False
            print("Learning Finished!! \n")
        
        #condition 3: condition is not best, but not yet finish.
        else:
            count += 1

    print("Training Accuracy : ", best_train_acc,  "Validation Accuracy : ", best_val_acc)

    saver.restore(sess, save_path)

    test_h, test_p, test_a = sess.run([hypothesis, predicted, accuracy],
                                      feed_dict={X: test_x, Y: test_y, keep_prob:1.0 , phase:False})
    print("\nTest Accuracy: ", test_a)
    best_test_acc = test_a


    model_num += 1                    

data_x, data_y = dp.divide_xy_test(raw_data, 'Platinum_Status', 1,-2)
ids = raw_data['patient']

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, './model0.ckpt')
    test_h, test_p = sess.run([hypothesis, predicted], feed_dict={X: data_x, Y: dp.one_hot_encoder(data_y), keep_prob:1.0 , phase:False})

dp.prediction_probability(test_h, test_p, data_y, ids)


# In[12]:


len(test_h)


# In[22]:


len(data_y)


# In[23]:


len(ids)


# In[24]:


len(test_p)


# In[25]:


test_p


# In[26]:


test_h


# In[28]:




