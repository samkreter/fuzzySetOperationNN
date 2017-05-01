import os
import sys


tests = ["100","20"]
regs = ['reg','noreg']



for test in tests:
    for reg in regs:
        os.system("python3 dnn.py " + reg + " combinedmul " + test)

    print("DONE!")