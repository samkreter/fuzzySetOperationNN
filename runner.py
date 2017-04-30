import os
import sys


tests = ["add","sub","combined"]
regs = ['reg','noreg']



for test in tests:
    for reg in regs:
        os.system("python3 dnn.py " + reg + " " + test)