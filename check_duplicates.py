import os
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import numpy as np
import scipy
import pandas as pd
import re
import math
import seaborn as sns


def checkIfDuplicates_1(listOfElems):
    ''' Check if given list contains any duplicates '''    
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True

def load_pandas(file_name):
    dframe = pd.read_csv(file_name)
##read date/time
    date = dframe['Date/Depth']
    date = np.array(date)
    #print(type(date))
    result = checkIfDuplicates_1(date)
    
    if result:
    	print("Loading file: ", file_name)
    	print('Duplicates Found')
    	
def main():

	from pathlib import Path
	directory = "/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/site_level/"
	directory_as_str = str(directory)
	pathlist = Path(directory_as_str).glob('*.csv')
	for path in pathlist:
	#because path is object not string
		path_in_str = str(path)
		pattern = re.compile(r'\d+')
		substring = pattern.findall(path_in_str)
		#print(substring)
		#print(path_in_str)
		boreholes = path_in_str
		load_pandas(boreholes)
		

main()

