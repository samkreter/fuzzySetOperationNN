from .memfuncs import MemFunc
import matplotlib.pyplot as plt
import numpy as np



#W value for Yager Operator
w = 2

#Base level for creating fuzzy numbers
INC = .1

#Used for printing the workings of the experiements
show = False

#Graphing constant
linewidth = 5


#Base class for computer fuzzy number operations using the extention princple
class ExtensionOps:

    def __init__(self,op):

        #Convert the string input to an actual operator, allows for extention
        opDict = {'add':np.add,
                  'sub':np.subtract,
                  'div':np.divide,
                  'mul':np.multiply,
                  'max':max,
                  'min':min,
                  'pow':pow}

        #String Name of the operator
        self.opName = op

        #Weird worka round for the complemnt
        if op == "comp":
            self.func = self.comp
        elif op in opDict:
            self.op = opDict[op]
            self.func = self.extention
        else:
            raise ValueError("Invalid Operator")


    #Convert the domain from 3 letter value into full dsicreate domain
    def convertToDomain(self,A):


        mem1 = MemFunc("trap",A)
        newA = []

        for i in np.arange(0,1.05,.05):

            newA.append([i,self.round2(mem1.memFunc(i))])


        return newA

    #Used to fixed the numpy rounding errors
    def round2(self,val):
        val = int(val * 100)
        return (val / 100)

    #More issues with rounding
    def round_to_05(self,n):
        correction = 0.5 if n >= 0 else -0.5
        return int( n/.05+correction ) * .05


    #The compliment operator for the extention principle
    def comp(self,A):

        A = A[0]

        #If there are only 4 elements, convert the domain
        if len(A) == 4:
            A = self.convertToDomain(A)

        #The ouput varible for the domain
        out = [[],[]]

        for a in A:
            z = self.round_to_05(self.round2(1.0 - a[0]))
            f = a[1]

            try:
                index = out[0].index(z)
                out[1][index] = max(out[1][index],f)
            except ValueError:
                out[0].append(z)
                out[1].append(f)


        #Combine and sort the output
        out = list(zip(out[0],out[1]))
        out.sort(key=lambda x:x[0])
        out1 = list(zip(*out))

        #Convert the array back to a numpy array for graphing
        A = np.array(A)

        #Graph the operation that had just been computed
        if show:
            plt.title("Compliment")
            l1, = plt.plot(out1[0],out1[1],c='y',linewidth=linewidth,label="Output")
            plt.plot(A[:,0],A[:,1],c='k',linewidth=linewidth)
            plt.xlim([0,1])
            plt.legend(handles=[l1])
            plt.ylim([0,1])
            plt.show()

        return out


    # Base function for creating computer the operation
    # \params: N fuzzy numbers to use the operation on
    # \return: fuzzy number with the answer
    def extention(self, params):


        A = params[0]
        for i in range(1,len(params)):
            B = params[i]

            #Convert a membership function to the right domain the first time
            if len(A) == 4:
                A = self.convertToDomain(A)
            if len(B) == 4:
                B = self.convertToDomain(B)


            out = [[],[]]

            for a in A:
                for b in B:
                    z = self.round2(self.op(a[0], b[0]))

                    #Make sure there is a value involved
                    try:
                        b[1]
                    except:
                        continue

                    f = min(a[1],b[1])

                    try:
                        index = out[0].index(z)
                        out[1][index] = max(out[1][index],f)
                    except ValueError:
                        out[0].append(z)
                        out[1].append(f)



            #Combine and sort the output
            out = list(zip(out[0],out[1]))
            out.sort(key=lambda x:x[0])

            #Convert to numpy array to graph
            B = np.array(B)
            A = np.array(A)

            #Fun graphing section
            if show:
                out1 = np.array(list(zip(*out)))
                plt.plot(A[:,0],A[:,1],c='b',linewidth=linewidth)
                plt.plot(B[:,0],B[:,1],c='r',linewidth=linewidth)
                l1, = plt.plot(out1[0],out1[1],c='y',linewidth=linewidth, label="Output")
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.legend(handles=[l1])
                plt.title(self.opName)
                plt.show()

            A = out


        return A




# e = ExtentionOps("add")
# mem1 = MemFunc('tri',[.2,.2,.4])
# mem2 = MemFunc('tri',[.4,.6,.8])
# #mem2 = lambda x: 1 if x == 1 else 0



# A = []
# B = []

# for i in np.arange(0,1,.05):

#     A.append([i,e.round2(mem1.memFunc(i))])

#     B.append([i,e.round2(mem2.memFunc(i))])

# A = np.array(A)
# B = np.array(B)



# #A = [.2,.4,.4,.6]
# # B = [.4,.6,.6,.8]

# print("########")
# #p = e.comp(A)
# t = e.extention([A,B])
# #print(e.extention([p,t]))






