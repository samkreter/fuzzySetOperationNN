import os
import sys



tests = ['combinedmul']




for test in tests:
    os.system("python3 dnn.py nonreg " +  test + " 20")
    print("DONE!")
