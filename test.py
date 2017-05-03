
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random

from SamsCI import *


def show_result(pred,truth,title):



    #pred = list(map(round,pred))
    #truth = list(map(round,truth))

    print(pred)
    print(truth)

    #m_pred = MemFunc('trap',pred)
    #m_truth = MemFunc('trap',truth)



    X = np.arange(-5,20, .5)

    plt.plot(pred,[0,1,1,0],c='#31698a',linewidth=4)
    plt.plot(truth,[0,1,1,0], c='#cc0000',linewidth=4)

    #plt.xlim([-3,3])
    plt.ylim([0,1])
    #plt.legend(handles=[l1])
    plt.title(title)
    plt.show()


dirs = ["./first_run_tests","./second_run_tests","./third_run_tests"]

i = 0
res = {}
for data_dir in dirs:
    for file in os.listdir(data_dir):
        if file.endswith(".pickle"):
            with open(data_dir + "/" + file,"rb") as f:
                res[file.split('.')[0] + str(i)] = pickle.load(f)
            i += 1


for key,val in res.items():
    print(key)

    ran = random.randint(0,len(val[0]) - 1)
    truth = [1]


    while len(set(truth)) == 1:
        pred = [round(i,2) for i in val[0][ran]]
        truth = [round(i,2) for i in val[1][ran]]

        ran = random.randint(0,len(val[0]) - 1)


    show_result(pred,truth,key)





