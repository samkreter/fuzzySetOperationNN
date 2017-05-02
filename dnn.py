import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import random
import pickle
import os
import sys

#All my CI code that i've written
from SamsCI import *


DATA_FOLDER = "data/"


REGUALIZE = True if sys.argv[1] == "reg" else False


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
        'add':[0,0],
        'sub':[1,0],
        'mul':[0,1],
        'div':[1,1]
    }


    if featureOp:
        add_feature = f_op_map[op]
    else:
        add_feature = []

    try:
        with open(DATA_FOLDER + filename + "_" + op + "_" + str(featureOp) + ".pickle",'rb') as f:
            samples,labels = pickle.load(f)
            print("Reading: " + filename + "_" + op + "_" + str(featureOp) + ".pickle")

    except:
        print("Generating: " + filename + "_" + op + "_" + str(featureOp) + ".pickle")
        samples = []
        labels = []
        alpha = AlphaOps(op).alphaCuts
        bs = random.sample(list(np.arange(0,10,.01)),500)


        for b1 in bs:
            # if b1 == 0:
            #     continue

            for b2 in bs:

                # if b2 == 0:
                #     continue



                A = list(map(s_round,[b1 - wideValue, b1, b1, b1 + wideValue]))
                B = list(map(s_round,[b2 - wideValue, b2, b2, b2 + wideValue]))


                label = list(map(s_round,alpha([A,B])))
                samples.append(add_feature + A + B)
                labels.append(label)



        with open(DATA_FOLDER + filename + "_" + op + "_" + str(featureOp) + ".pickle",'wb') as f:
            pickle.dump((samples,labels),f)

    return samples,labels


def create_combined(data):
    X = []
    y = []

    for el in data:
        X += el[0]
        y += el[1]


    combined = list(zip(X,y))
    random.shuffle(combined)

    X,y = zip(*combined)

    print(X[0])
    print(y[0])

    return X,y


if sys.argv[2] == "combined":
    X_sub_feat,y_sub_feat = generate_training_full(op='sub',featureOp=True)
    X_add_feat,y_add_feat = generate_training_full(op='add',featureOp=True)
    X,y = create_combined(X_sub_feat,y_sub_feat,X_add_feat,y_add_feat)
elif sys.argv[2] == "combinedmul":
    data1 = generate_training_full(op='sub',featureOp=True)
    data2 = generate_training_full(op='add',featureOp=True)
    data3 = generate_training_full(op='mul',featureOp=True)
    data4 = generate_training_full(op='div',featureOp=True)
    X,y = create_combined([data1,data2,data3,data4])
elif sys.argv[2] == "div":
    X,y = generate_training_full(op='div',featureOp=False)
elif sys.argv[2] == "mul":
    X,y = generate_training_full(op='mul',featureOp=False)
elif sys.argv[2] == "sub":
    X,y = generate_training_full(op='sub',featureOp=False)
else:
    X,y = generate_training_full(op='add',featureOp=False)



train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=.2,random_state = 1)


n_nodes_hl1 = 20
n_nodes_hl2 = 20
n_nodes_hl3 = 20

n_classes = 4


x = tf.placeholder('float',[None,len(train_x[0])])
y = tf.placeholder('float',[None,4])

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




def train_network(x):
    pred = neural_net_model(x)
    cost = tf.reduce_sum(tf.pow(pred - y,2))/(len(train_y))


    beta = .01

    if REGUALIZE:
        cost = tf.reduce_mean(cost +
            beta*tf.nn.l2_loss(weights['h1_layer']) +
            beta*tf.nn.l2_loss(weights['h2_layer']) +
            beta*tf.nn.l2_loss(weights['output_layer']))


    file = File(sys.argv[1] + "_" + sys.argv[2] + "_" + sys.argv[3] + "_log.csv")
    saver = tf.train.Saver()

    optimizer = tf.train.AdamOptimizer().minimize(cost)



    n_epochs = 5000
    printer = 10

    errors = []

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())



        try:
            with open("epoch.log","rb") as f:
                epoch = pickle.load(f)
        except:
            epoch = 0


        if epoch > 0:
            saver.restore(sess,"./model_" + sys.argv[1] + "_" + sys.argv[2]  + "_" + sys.argv[3]+ ".ckpt")


        print("Starting: " + sys.argv[1] + "_" + sys.argv[2]+ "_" + sys.argv[3])

        while epoch < n_epochs:


            _,c = sess.run([optimizer,cost],feed_dict = {x: train_x, y: train_y})



            #if epoch % printer == 0:
                #print(sess.run(pred,feed_dict={x:[test]}))
            preds = sess.run(pred, feed_dict={x:test_x})
            accuracy = getAccuarcy(preds,test_y)
            print(accuracy)
            file.writeA([c,accuracy])


            print("Epoch:",epoch,"completed out of:", n_epochs, "Loss:", c)

            #save the epoch we are currently on
            with open("epoch.log","wb") as f:
                pickle.dump(epoch,f)
            #errors.append(c)
            saver.save(sess,"model_" + sys.argv[1] + "_" + sys.argv[2]+ "_" + sys.argv[3] + ".ckpt")
            epoch += 1

        os.remove("epoch.log")

        #print(sess.run(pred,feed_dict = {x:[test]}))



def getAccuarcy(preds,truths):
    correct = 0
    for pred,truth in zip(preds,truths):

        pred = list(map(round,pred))
        truth = list(map(round,truth))
        if pred == truth:
            correct += 1
        print(pred)
        print(truth)
        t = input()
    return (correct / len(preds)) * 100


def test_network():

    pred = neural_net_model(x)

    saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        #saver.restore(sess,"model_" + sys.argv[1] + "_" + sys.argv[2]  + "_" + sys.argv[3]+ ".ckpt")
        saver.restore(sess,"./model_nonreg_div_20.ckpt")


        preds = sess.run(pred, feed_dict={x:test_x})
   

        accuracy = getAccuarcy(preds,test_y)
        print(accuracy)
        # all_vars = tf.get_collection('vars')
        # for v in all_vars:
        #     v_ = sess.run(v)
        #     print(v_)



if __name__ == '__main__':

    #train_network(x)
    test_network()


