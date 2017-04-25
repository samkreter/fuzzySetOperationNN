import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import random

SAVED = False


def s_round(a):
    return int((a * 10) + 0.5) / 10.0

def generate_training_10(wideValue=.1):
    samples = []
    labels = []
    fNumbers = []

    #compute values
    for b in np.arange(0,1.1,.1):
        a = s_round(max(0,b - wideValue))
        c = s_round(min(1,b + wideValue))
        b = s_round(b)
        fNumbers.append([a,b,b,c])


    #Compute pairs
    for A in fNumbers:
        for B in fNumbers:
            samples.append(A + B)
            labels.append([min(1,s_round(i[0] + i[1])) for i in zip(A,B) ])

    return samples,labels

def generate_training_full(wideValue=2):
    samples = []
    labels = []

    bs = random.sample(range(0,100),100)

    for b1 in bs:
        for b2 in bs:
            b = b1 + b2
            samples.append([b1 - wideValue, b1, b1, b1 + wideValue, b2 - wideValue, b2, b2, b2+ wideValue])
            labels.append([b - wideValue, b, b, b + wideValue])
    return samples,labels


X1,y1 = generate_training_full()
train_x, test_x, train_y, test_y = train_test_split(X1,y1,test_size=.2,random_state = 1)


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 4


# x = tf.placeholder('float',[None,len(train_x[0])])
# y = tf.placeholder('float',[None,4])

x = tf.placeholder('float')
y = tf.placeholder('float')


def neural_net_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                  'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    #l1 = tf.nn.relu(l1)
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    #l2 = tf.nn.relu(l2)
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    #l3 = tf.nn.relu(l3)
    l3 = tf.nn.sigmoid(l3)


    output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])


    return output



def train_network(x):
    pred = neural_net_model(x)
    cost = tf.reduce_sum(tf.pow(pred - y,2))/(len(train_y))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))

    saver = tf.train.Saver()

    optimizer = tf.train.AdamOptimizer().minimize(cost)


    n_epochs = 700
    printer = 20

    errors = []

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        test = np.array([1,2,2,3,1,2,2,3])

        #print(sess.run(pred,feed_dict={x:[test]}))

        if SAVED:
            saver.restore(sess,"model.ckpt")
        else:
            for epoch in range(n_epochs):

                if epoch % printer == 0:
                    print(sess.run(pred,feed_dict={x:[test]}))

                i = 0

                _,c = sess.run([optimizer,cost],feed_dict = {x: train_x, y: train_y})

                print("Epoch:",epoch,"completed out of:", n_epochs, "Loss:", c)
                errors.append(c)
                saver.save(sess,"model.ckpt")




        print(sess.run(pred,feed_dict = {x:[test]}))
    plt.plot(range(n_epochs),errors)
    plt.show()


train_network(x)


