import os
import sys


tests = ["20"]

regs = ['div','combinedmul']



for test in tests:
    for reg in regs:
        os.system("python3 dnn.py nonreg " +  reg + " " + test)

    print("DONE!")
