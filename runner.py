import os
import sys



tests = ['add','sub','mul','div']




for test in tests:
    for reg in regs:
        os.system("python3 dnn.py nonreg " +  test + " 20")

    print("DONE!")
