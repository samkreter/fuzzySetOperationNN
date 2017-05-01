import os
import sys


tests = ["100","20"]

regs = ['mul','div','combinedmul']



for test in tests:
    for reg in regs:
        os.system("python3 dnn.py nonreg " +  reg + " " + test)

    print("DONE!")