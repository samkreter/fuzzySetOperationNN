import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import random
import pickle
import os

#All my CI code that i've written
from SamsCI import *

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

def generate_training_full(wideValue=2,op="add"):
    samples = []
    labels = []
    alpha = AlphaOps(op).alphaCuts
    bs = random.sample(list(np.arange(0,10,.01)),500)


    for b1 in bs:
        for b2 in bs:
            A = list(map(s_round,[b1 - wideValue, b1, b1, b1 + wideValue]))
            B = list(map(s_round,[b2 - wideValue, b2, b2, b2 + wideValue]))
            label = list(map(s_round,alpha([A,B])))

            samples.append(A + B)
            labels.append(label)
    return samples,labels


X1,y1 = generate_training_full()


train_x, test_x, train_y, test_y = train_test_split(X1,y1,test_size=.2,random_state = 1)


n_nodes_hl1 = 20
n_nodes_hl2 = 20
n_nodes_hl3 = 20

n_classes = 4


# x = tf.placeholder('float',[None,len(train_x[0])])
# y = tf.placeholder('float',[None,4])

x = tf.placeholder('float')
y = tf.placeholder('float')

weights = {
    'h1_layer': tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])),
    'h2_layer': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
    'output_layer': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes]))
}

biases = {
    'h1_layer': tf.Variable(tf.random_normal([n_nodes_hl1])),
    'h2_layer': tf.Variable(tf.random_normal([n_nodes_hl2])),
    'output_layer': tf.Variable(tf.random_normal([n_classes]))
}

def neural_net_model(data):

    l1 = tf.add(tf.matmul(data,weights['h1_layer']), biases['h1_layer'])
    l1 = tf.nn.tanh(l1)

    l2 = tf.add(tf.matmul(l1,weights['h2_layer']), biases['h2_layer'])
    l2 = tf.nn.tanh(l2)

    output = tf.add(tf.matmul(l2,weights['output_layer']), biases['output_layer'])

    return output


beta = .01

def train_network(x):
    pred = neural_net_model(x)
    cost = tf.reduce_sum(tf.pow(pred - y,2))/(len(train_y))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))


    cost = tf.reduce_mean(cost +
        beta*tf.nn.l2_loss(weights['h1_layer']) +
        beta*tf.nn.l2_loss(weights['h2_layer']) +
        beta*tf.nn.l2_loss(weights['output_layer']))



    saver = tf.train.Saver()

    optimizer = tf.train.AdamOptimizer().minimize(cost)



    n_epochs = 1000000
    printer = 20

    errors = []

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        test = np.array([5,6,6,7,1,2,2,3])
        #test = np.array([.1,.2,.2,.3,.1,.2,.2,.3])

        #print(sess.run(pred,feed_dict={x:[test]}))


        try:
            with open("epoch.log","rb") as f:
                epoch = pickle.load(f)
        except:
            epoch = 0


        if epoch > 0:
            saver.restore(sess,"model.ckpt")

        while epoch < n_epochs:

            if epoch % printer == 0:
                print(sess.run(pred,feed_dict={x:[test]}))


            _,c = sess.run([optimizer,cost],feed_dict = {x: train_x, y: train_y})

            print("Epoch:",epoch,"completed out of:", n_epochs, "Loss:", c)

            #save the epoch we are currently on
            with open("epoch.log","wb") as f:
                pickle.dump(epoch,f)
            #errors.append(c)
            saver.save(sess,"model.ckpt")
            epoch += 1

        os.remove("epoch.log")
        print(sess.run(pred,feed_dict = {x:[test]}))


train_network(x)


