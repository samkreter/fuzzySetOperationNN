import matplotlib.pyplot as plt
import numpy as np
from .MemFuncs import MemFunc

#W value for the Yager operators
w = 2
#Constant for graphing
show = False


#Base class for level cuts operations
class AlphaOps:


    def __init__(self,op):
        #String holding the naem of the operations
        self.opName = op
        #Use the string to retrieve the operation function
        self.op = getattr(self,op)
            #raise ValueError("Do not have that operator in AlphaOps")

    #Convert membership representation into A for interval [A,B]
    def pos1(self, fNum, alpha):
        return ((fNum[1]-fNum[0]) * alpha + fNum[0])

    #Convert membership representation into B for interval [A,B]
    def pos2(self, fNum, alpha):
        size = len(fNum)
        return (fNum[size - 1] - (fNum[size - 1]-fNum[size - 2]) * alpha)

    ######### Base operators ############
    def add(self,a,b,c,d):
       return [a + c, b + d]

    def sub(self,a,b,c,d):
        return [a - d, b - c]

    def max(self,a,b,c,d):
        return [max(a,c), max(b,d)]

    def min(self,a,b,c,d):
        return [min(a,c), min(b,d)]

    def mul(self,a,b,c,d):
        return [min(a*c,a*d,b*c,b*d), max(a*c,a*d,b*c,b*d)]

    def div(self,a,b,c,d):
        if d != 0:
            d = 1 / d

        if c != 0:
            c = 1 / d

        return self.mul(a,b,d,c)

    #Base compliment operator
    def comp(self,a,b):
        return [1-b, 1-a]


    #Function for computing all operations
    def alphaCuts(self, params):

        #The levels of alpha cuts to take
        alphas = [0,.2,.8,1]

        fNum1 = params[0]

        #Create the domain for graphing
        X = np.arange(0,1.05, .05)

        points = []

        #If only one param then do the complement / special case
        if len(params) == 1:
            #Computer the interval operations for each of the selected alphas
            for alpha in alphas:
                a = self.pos1(fNum1,alpha)
                b = self.pos2(fNum1,alpha)

                points.append(self.op(a,b))

            #Compute the membership function for graphing
            m1 = MemFunc('trap',fNum1)

            #Convert the point values from the alphas into a 3 point membership function
            fNum1 = [points[0][0],points[3][0],points[3][1],points[0][1]]

            print(points)

            #Graph the resluts of the operation
            if(show):
                m2 = MemFunc('trap',fNum1)
                plt.plot(X,[m1.memFunc(i) for i in X ],c='k',linewidth=4)
                plt.plot(X,[m2.memFunc(i) for i in X], c='y',linewidth=4)
                plt.xlim([0,1])
                plt.ylim([0,1])
                #plt.legend(handles=[l1])
                plt.title(self.opName)
                plt.show()

            return fNum1




        #Use the belive equations to get the alpha intervals from the membership functions
        #TRI: [(b-a)alpha + a, c - (c-b)alpha]
        #TRAP: [(b-a)alpha + a, d - (d-c)alpha]

        for i in range(1,len(params)):
            fNum2 = params[i]
            for alpha in alphas:

                a = self.pos1(fNum1,alpha)
                b = self.pos2(fNum1,alpha)
                c = self.pos1(fNum2,alpha)
                d = self.pos2(fNum2,alpha)

                points.append(self.op(a,b,c,d))


            m1 = MemFunc('trap',fNum1)
            m2 = MemFunc('trap',fNum2)

            #Create a trap membership function, tri is the same but the b and c values are equal
            #Comment out for regular fuzzy sets
            fNum1 = [points[0][0],points[3][0],points[3][1],points[0][1]]


            points = []

            mOut = MemFunc('trap',fNum1)

            #Graph the sets
            if show:
                plt.plot(X,[m1.memFunc(i) for i in X ],c='k',linewidth=4)
                plt.plot(X,[m2.memFunc(i) for i in X], c='b',linewidth=4)
                plt.plot(X,[mOut.memFunc(i) for i in X], c='y',linewidth=4)
                plt.xlim([0,1])
                plt.ylim([0,1])
                #plt.legend(handles=[l1])
                plt.title(self.opName)
                plt.show()


        return fNum1

# a = AlphaOps("add").alphaCuts
# A = [8.56,10,10,12.45]
# B = [8,11, 11 ,12]

# f = a([A,B])


# m1 = MemFunc('tri',A)
# m2 = MemFunc('tri',B)
# f1 = MemFunc('trap',f)

# X = np.arange(0,2,.1)


# print([f1.memFunc(i) for i in X ])

# plt.plot(X,[m1.memFunc(i) for i in X ],c='g')
# plt.plot(X,[m2.memFunc(i) for i in X ],c='b')
# plt.plot(X,[f1.memFunc(i) for i in X ],c='r')
# plt.show()






