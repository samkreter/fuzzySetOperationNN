
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

#All my CI code that i've written
from SamsCI import *




#TODO
# FInd different ways to represnet Alpha cuts in the Neural Net

a = AlphaOps("add").alphaCuts

A = [.2,.5,.7]
B = [.3,.5,.7]

f = a([A,B])



def s_round(a):
    return int((a * 10) + 0.5) / 10.0

def generate_training_full(wideValue=2):
    samples = []
    labels = []

    bs = random.sample(range(0,5),5)

    for b1 in bs:
        for b2 in bs:
            b = b1 + b2
            samples.append([b1 - wideValue, b1, b1, b1 + wideValue, b2 - wideValue, b2, b2, b2+ wideValue])
            labels.append([b - wideValue, b, b, b + wideValue])
    return samples,labels

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


def getError(errors):
    length = len(errors)
    errorSum = 0.0
    errorList = [0,0,0]
    for i in range(length):
        if errors[i][0] != errors[i][1]:
            errorSum += 1
    return errorSum / length




def full_case(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state = 1)


    nn = NueralNet(learningRate=.8,activFunc='linear',outputActivFunc='linear',numInputs=8,numOutputs=4,numHLayers=2,numHiddenNodes=[3,3,3,3,3,3], hBias=0.35, outputBias=0.6, outWeights=[0.4, 0.45, 0.5, 0.55])

    main_errors = []

    for i in range(300):
        srange = list(range(len(X_train)))
        np.random.shuffle(srange)
        for j in srange:
            nn.train(X_train[j], y_train[j])
        errors = []
        trange = list(range(len(X_test)))
        np.random.shuffle(trange)
        for k in trange:
            preds = nn.predict(X_test[k])

            errors.append([y_test[k],preds])

        print(errors)
        break



    plt.plot(range(len(main_errors)),main_errors)
    plt.show()
    #clf.fit(X_train,y_train)
    #preds = clf.predict(X_test)
    #print("%.2f" % (accuracy_score(preds,y_test) * 100))



def base_case(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state = 1)


    nn = NueralNet(learningRate=.8,activFunc='sigmoid',outputActivFunc='sigmoid',numInputs=8,numOutputs=4,numHLayers=2,numHiddenNodes=[3,3,3,3,3,3], hBias=0.35, outputBias=0.6, outWeights=[0.4, 0.45, 0.5, 0.55])

    main_errors = []

    for i in range(300):
        srange = list(range(len(X_train)))
        np.random.shuffle(srange)
        for j in srange:
            nn.train(X_train[j], y_train[j])
        errors = []
        trange = list(range(len(X_test)))
        np.random.shuffle(trange)
        for k in trange:
            preds = list(map(lambda x: s_round(x),nn.predict(X_test[k])))

            errors.append([y_test[k],preds])

        #print(errors)
        print(getError(errors))
        main_errors.append(getError(errors))


    plt.plot(range(len(main_errors)),main_errors)
    plt.show()
    #clf.fit(X_train,y_train)
    #preds = clf.predict(X_test)
    #print("%.2f" % (accuracy_score(preds,y_test) * 100))


if __name__ == '__main__':
   X, y = generate_training_10()
   X1,y1 = generate_training_full()
   print(y1)
   full_case(X1,y1)

# m1 = MemFunc('tri',A)
# m2 = MemFunc('tri',B)
# f1 = MemFunc('trap',f)

# X = np.arange(0,2,.1)


# print([f1.memFunc(i) for i in X ])

# plt.plot(X,[m1.memFunc(i) for i in X ],c='g')
# plt.plot(X,[m2.memFunc(i) for i in X ],c='b')
# plt.plot(X,[f1.memFunc(i) for i in X ],c='r')
# plt.show()
