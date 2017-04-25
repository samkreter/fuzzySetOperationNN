import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import random


def generate_training_full(wideValue=2):
    samples = []
    labels = []

    bs = random.sample(range(0,1000),10)

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


x = tf.placeholder('float',[None,len(train_x[0])])
y = tf.placeholder('float',[None,4])


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
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)


    output = tf.add(tf.matmul(l3,output_layer['weights']), output_layer['biases'])


    return output

saver = tf.train.Saver()

def train_network(x):
    pred = neural_net_model(x)
    cost = tf.reduce_sum(tf.pow(pred - y,2))/(len(train_y))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)


    n_epochs = 40

    errors = []

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(n_epochs):
            epoch_loss = 0

            i = 0

            _,c = sess.run([optimizer,cost],feed_dict = {x: train_x, y: train_y})

            print("Epoch:",epoch,"completed out of:", n_epochs, "Loss:", c)
            errors.append(c)


    plt.plot(range(n_epochs),errors)
    #plt.show()
    pred.eval(feed_dict = {x:[1,2,2,3,1,2,2,3]})

train_network(x)


