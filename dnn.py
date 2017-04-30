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

def generate_training_full(wideValue=2,op="add",force=False,filename='gen_data',featureOp=False):

    f_op_map = {
        'add':[1],
        'sub':[0]
    }


    if featureOp:
        add_feature = f_op_map[op]
    else:
        add_feature = []

    try:
        with open(filename + "_" + op + "_" + str(featureOp) + ".pickle",'rb') as f:
            samples,labels = pickle.load(f)
            print("Reading: " + filename + "_" + op + "_" + str(featureOp) + ".pickle")

    except:
        print("Generating: " + filename + "_" + op + "_" + str(featureOp) + ".pickle")
        samples = []
        labels = []
        alpha = AlphaOps(op).alphaCuts
        bs = random.sample(list(np.arange(0,10,.01)),500)


        for b1 in bs:
            for b2 in bs:
                A = list(map(s_round,[b1 - wideValue, b1, b1, b1 + wideValue]))
                B = list(map(s_round,[b2 - wideValue, b2, b2, b2 + wideValue]))
                label = list(map(s_round,alpha([A,B])))

                samples.append(add_feature + A + B)
                labels.append(label)

        with open(filename + "_" + op + "_" + str(featureOp) + ".pickle",'wb') as f:
            pickle.dump((samples,labels),f)

    return samples,labels


def create_combined(X1,y1,X2,y2):
    X = X1 + X2
    y = y1 + y2

    combined = list(zip(X,y))
    random.shuffle(combined)

    X,y = zip(*combined)

    return X,y



X1,y1 = generate_training_full(op='sub',featureOp=True)
X2,y2 = generate_training_full(op='add',featureOp=True)


X,y = create_combined(X1,y1,X2,y2)


train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=.2,random_state = 1)


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
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1,weights['h2_layer']), biases['h2_layer'])
    l2 = tf.nn.sigmoid(l2)

    output = tf.add(tf.matmul(l2,weights['output_layer']), biases['output_layer'])

    return output


beta = .01

def train_network(x):
    pred = neural_net_model(x)
    cost = tf.reduce_sum(tf.pow(pred - y,2))/(len(train_y))



    # cost = tf.reduce_mean(cost +
    #     beta*tf.nn.l2_loss(weights['h1_layer']) +
    #     beta*tf.nn.l2_loss(weights['h2_layer']) +
    #     beta*tf.nn.l2_loss(weights['output_layer']))


    file = File('logs.csv')
    saver = tf.train.Saver()

    optimizer = tf.train.AdamOptimizer().minimize(cost)



    n_epochs = 1000000
    printer = 20

    errors = []

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        test = np.array([1,5,6,6,7,1,2,2,3])
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


            _,c = sess.run([optimizer,cost],feed_dict = {x: train_x, y: train_y})



            if epoch % printer == 0:
                print(sess.run(pred,feed_dict={x:[test]}))
                preds = sess.run(pred, feed_dict={x:test_x})
                accuracy = getAccuarcy(preds,test_y)
                print(accuracy)
                file.writeA([c,accuracy])


            print("Epoch:",epoch,"completed out of:", n_epochs, "Loss:", c)

            #save the epoch we are currently on
            with open("epoch.log","wb") as f:
                pickle.dump(epoch,f)
            #errors.append(c)
            saver.save(sess,"model.ckpt")
            epoch += 1

        os.remove("epoch.log")

        print(sess.run(pred,feed_dict = {x:[test]}))



def getAccuarcy(preds,truths):
    correct = 0
    for pred,truth in zip(preds,truths):
        pred = list(map(round,pred))
        truth = list(map(round,truth))
        if pred == truth:
            correct += 1

    return (correct / len(preds)) * 100


def test_network(x):
    saver = tf.train.Saver()


    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())

        saver.restore(sess,"model.ckpt")

        all_vars = tf.get_collection('vars')
        for v in all_vars:
            v_ = sess.run(v)
            print(v_)

        #preds = sess.run(pred, feed_dict={x:test_x})

        #print(getAccuarcy(preds,test_y))

if __name__ == '__main__':
    train_network(x)
    #test_network(x)


