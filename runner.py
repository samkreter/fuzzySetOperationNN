import os
import sys



tests = ['combined']
regs = ['noreg','reg']




for test in tests:
    for reg in regs:
        os.system("python3 dnn.py " + reg + " " +  test)
        print("DONE!")
