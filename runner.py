import os
import sys



tests = ['add','sub','mul','div']




for test in tests:
    os.system("python3 dnn.py nonreg " +  test + " 20")
    print("DONE!")
