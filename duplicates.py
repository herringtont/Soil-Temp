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


fname = "/mnt/data/users/herringtont/soil_temp/In-Situ/Master_Lat_Long_Obs.csv"
dfil = "/mnt/data/users/herringtont/soil_temp/In-Situ/dupes.csv"
#read data into pandas
dframe = pd.read_csv(fname)
lat = dframe['Lat']
lon = dframe['Long']
site = dframe['Site_ID']
dataset = dframe['Dataset']
dframe = dframe.set_index(dframe['Site_ID'])
dupes = pd.concat(g for _, g in dframe.groupby("Lat") if len(g) > 1)
#print(dupes)
dupes2 = pd.concat(h for _, h in dupes.groupby("Long") if len(h) > 1)
#print(dupes2)
dupes3 = dupes2[dupes2.duplicated(['Lat', 'Long'])]
#print(dupes3)
#dupes2.to_csv(dfil, index=False)
dupes4 = pd.concat( i for _, i in dupes3.groupby("Dataset") if len(i) > 1)
#print(dupes4)
Krop = dupes4.loc[dupes4['Dataset'] == "Kropp"]
GTN = dupes4.loc[dupes4['Dataset'] == "GTN-P"]

Krop_ll = Krop[['Lat','Long','Site_ID']]
GTN_ll = GTN[['Lat', 'Long','Site_ID']]
dupes5 = set(Krop_ll).intersection(GTN_ll)
#print(Krop_ll)
#print(GTN_ll)
lat_un = np.unique(GTN_ll['Lat'])
lon_un = np.unique(GTN_ll['Long'])

dup_val = []
for j in lat_un:
    for k in lon_un: 
    	valK = Krop_ll.loc[(Krop_ll['Lat'] == j) & (Krop_ll['Long'] == k)]
    	sitid = valK['Site_ID']
    	if (valK.empty):
    		continue
    	
    	dup_val.append(valK)
print(dup_val)
#dup_val2 = pd.DataFrame(data=dup_val)
#dup_val2.replace("", "NaN", inplace=True)
#dup_val2.dropna()
#print(dup_val2)
#dup_val2.to_csv(dfil)
