import os
import sys




regs = ['combinedmul']



for test in tests:
    for reg in regs:
        os.system("python3 dnn.py nonreg " +  reg)

    print("DONE!")
