import os

BASE_DIR = os.path.dirname(os.path.abspath(''))
#BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # directory of file
BASE2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # base directory
#WORK_DIR = os.getcwd() # current directory

print(BASE_DIR)
#print(WORK_DIR)
print(BASE2_DIR)