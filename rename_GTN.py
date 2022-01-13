# -*- coding: utf-8 -*-

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


#set directories and files
wkdir = str("/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/")
odir = str("/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/site_level/")
filnm = str("/mnt/data/users/herringtont/soil_temp/In-Situ/GTN-P/GTN_Master.csv")

#read in filenames
dframe = pd.read_csv(filnm)
#print(dframe)

fil = dframe['Filename']

#loop through files
for i in range (0,68):
	sitid = i + 1
	fil2 = fil.iloc[i]
	print(fil2)
	str_i = str(sitid)
	sitnm = ["site_", str_i, ".csv"]
	sitename = "".join(sitnm)
	print(sitename)
	os.system("cp " +str(wkdir)+str(fil2)+ " "+str(odir)+str(sitename))
	
