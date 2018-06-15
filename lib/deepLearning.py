import tensorflow as tf
import numpy as np


def set_model(x, y, nodes , learning_rate):
    tf.reset_default_graph()


    np.random.seed(777)
    keep_prob = tf.placeholder(tf.float32) #0.7 -> 70% 켜진 채로 30% 꺼진 채로
    weights = []
    bias = []
    hidden_layers = []

    X = tf.placeholder(tf.float32, [None, len(x[0])])
    Y = tf.placeholder(tf.float32, [None, len(y[0])])


    for i in range(len(nodes)):
        if i == 0:
            weights.append(tf.get_variable(shape=[len(x[0]),  nodes[i]], name='weight' + str(i), initializer=tf.contrib.layers.xavier_initializer()))
            bias.append(tf.Variable(tf.random_normal([ nodes[i]]), name='bias' + str(i)))
            hidden_layers.append(tf.nn.relu(tf.matmul( X,  weights[i]) +  bias[i]))
            hidden_layers[i] = tf.nn.dropout( hidden_layers[i], keep_prob= keep_prob)
        else:
             weights.append(tf.get_variable(shape=[ nodes[i - 1],  nodes[i]], name='weight' + str(i), initializer=tf.contrib.layers.xavier_initializer()))
             bias.append(tf.Variable(tf.random_normal([ nodes[i]]), name='bias' + str(i)))
             hidden_layers.append(tf.nn.relu(tf.matmul( hidden_layers[i - 1],  weights[i]) +  bias[i]))
             hidden_layers[i] = tf.nn.dropout( hidden_layers[i], keep_prob= keep_prob)

    i = len(nodes) - 1
    weights.append(tf.get_variable(shape=[ nodes[i], len(y[0])], name='weight' + str(i + 1),
                                    initializer=tf.contrib.layers.xavier_initializer()))
    bias.append(tf.Variable(tf.random_normal([len(y[0])]), name='bias' + str(i + 1)))
    logits = tf.matmul(hidden_layers[i],  weights[i + 1]) +  bias[i + 1]
    #### lable0, label1 ######
    ####  40,   -1 ######

    hypothesis = tf.nn.softmax(logits)
    ####   label0 ,label1 #####
    #### 0,   0.56,  0.44 #####



    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal( predicted, tf.argmax( Y, 1))
    accuracy = tf.reduce_mean(tf.cast( correct_prediction, dtype=tf.float32))

    return X , Y , weights, bias,hidden_layers, logits  , hypothesis, cost , train, predicted , correct_prediction , accuracy , keep_prob





def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size, activation_fn=None,scope=scope)

def dense_batch_relu(x, phase, size, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size,activation_fn=None,scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase,scope='bn')
        return tf.nn.relu(h2, 'relu')


def set_model_BN(x, y, nodes, learning_rate) :
    tf.reset_default_graph()

    np.random.seed(777)
    keep_prob = tf.placeholder(tf.float32)

    X = tf.placeholder(tf.float32, [None, len(x[0])])
    Y = tf.placeholder(tf.float32, [None, len(y[0])])
    phase = tf.placeholder(tf.bool, name='phase')
    layers = []

    h1 = dense_batch_relu(X, phase, nodes[0], 'layer1')
    h2 = dense_batch_relu(h1, phase, nodes[1],'layer2')
    h3 = dense_batch_relu(h2, phase, nodes[2],'layer3')
    h4 = dense_batch_relu(h2, phase, nodes[3],'layer4')
    layers.append(h1)
    layers.append(h2)
    layers.append(h3)
    layers.append(h4)
    logits = dense(h4, len(y[0]), 'logits')
    hypothesis = tf.nn.softmax(logits)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= Y))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy computation
    # True if hypothesis>0.5 else False

    predicted = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal( predicted, tf.argmax( Y, 1))
    accuracy = tf.reduce_mean(tf.cast( correct_prediction, dtype=tf.float32))

    return X , Y , layers, logits, phase  , hypothesis, cost , train, predicted , correct_prediction , accuracy , keep_prob