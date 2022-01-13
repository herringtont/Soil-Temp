import os
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpl_patches
import scipy
import cdo
from cdo import Cdo
import re
import math
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cftime
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


################### set variable names ##################
lyr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper']
olr = ['outliers','zscore','IQR']
bdir = '/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/'




for i in olr: #loop through outlier types
    olri = i
    for j in lyr: #loop through layers
    	lyrj = j
    	indir = "".join([bdir,olri,"/",lyrj,"/"])  
    	pathlist = Path(indir).glob('*.csv')
	
    	print(indir)
################### create master arrays #####################
    	site_master = []
    	stemp_master = []
    	date_master = []
    	lat_master = []
    	lon_master = []
    	num_layer_master = []	
################### loop through files within a layer directory ####################
    	for path in sorted(pathlist, key=lambda path: int(path.stem.split("site_")[1].split("*.csv")[0])):
    		fil = str(path)
    		#print(fil)
    		dframe = pd.read_csv(fil)    	
    		sitid = fil.split("site_")[1].split(".csv")[0]
    		#print(sitid)
    		dframe['Site ID'] = int(sitid)
    		siteid = dframe['Site ID'].tolist()
    		#print(dframe)
    		sitnam = "".join(["site_",sitid]) 
    		stemp = dframe['Layer_Avg'].tolist()
    		dtst = dframe['Dataset'].iloc[1]
    		dat = dframe['Date'].tolist()
    		if (dtst == 'GTN-P'):
    			datstr =  [datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S') for x in dat]
    		elif (dtst == 'Kropp'):
    			datstr = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dat]
    		lat = dframe['Lat'].tolist()
    		lon = dframe['Long'].tolist()
    		num_layer = dframe['Depths_Incl'].tolist()    				  	 
################### fill arrays ######################
    		site_master.append(siteid)
    		stemp_master.append(stemp)
    		date_master.append(datstr)
    		lat_master.append(lat)
    		lon_master.append(lon)
    		num_layer_master.append(num_layer)

################## flatten list to 1D ################
    	site_master_1D = []
    	for sublist in site_master:
    		for item in sublist:
    			site_master_1D.append(item)
    	stemp_master_1D = []
    	for sublist in stemp_master:
    		for item in sublist:
    			stemp_master_1D.append(item)
    	date_master_1D = []
    	for sublist in date_master:
    		for item in sublist:
    			date_master_1D.append(item)
    	lat_master_1D = []
    	for sublist in lat_master:
    		for item in sublist:
    			lat_master_1D.append(item)
    	lon_master_1D = []
    	for sublist in lon_master:
    		for item in sublist:
    			lon_master_1D.append(item)
    	num_layer_master_1D = []
    	for sublist in num_layer_master:
    		for item in sublist:
    			num_layer_master_1D.append(item)	
    	situq =  np.unique(site_master_1D)


################### create master dataframe with all sites for a given outlier type/layer combo ###############
    	dframe_master = pd.DataFrame(data=site_master_1D, columns=['Site ID'])
    	dframe_master['Date'] = date_master_1D
    	dframe_master['Soil Temp'] = stemp_master_1D
    	dframe_master['Lat'] = lat_master_1D
    	dframe_master['Long'] = lon_master_1D
    	dframe_master['Num of Depths'] = num_layer_master_1D	
    	#print(dframe_master)



################### multiplot ####################### 

    	#min_site = situq[0]
    	#max_site = situq[len(situq)-1]
    	#print(len(situq))

    	ymin = -45
    	ymax = 45
    	xmin = datetime.date(1990,1,1)
    	xmax = datetime.date(2020,1,1)

    		
##### number of figures required ######
#Layer 1 - 10 (24 figures each)
#Layer 2 - 8 (24 figures in fig 1-7, 15 figures in fig 8)
#Layer 3 - 6 (24 figures in fig 1-5, 12 figures in fig 6)
#Layer 4 - 4 (24 figures in fig 1-3, 1 figure in fig 4)
#Layer 5 - 1 (24 figures)

    	
    	numsit = len(situq) 
    	if(numsit%24 == 0):
    		numfig = int((numsit/24))
    		lastfig = 24
    	elif(numsit%24 != 0):
    		numfig = int((numsit//24)+1)
    		lastfig = int(numsit%24)    	
    	print(situq)
    	#print("number of figures: ",numfig)
    	#print("remainder:",numsit%24)		
############## create subplots #######################
    	for i in range (0,numfig): #number of figures depends on soil layer
    		fig = plt.figure()

    		fig,axs = plt.subplots(nrows = 6, ncols = 4, sharex='col', sharey = 'row',figsize=(24,16)) #create a figure with 6x4 subplots
		
################### grab data for each site to plot #######################

    		if (i < (numfig-1)):
    			mxrg = 25
    		elif (i == (numfig-1)):
    			mxrg = lastfig + 1
    		if (lastfig%4 == 0): #if there is no remainder
    			numrow = int(lastfig/4)
    		elif (lastfig%4 != 0):
    			numrow = int((lastfig//4)+1)
    		#print("num row =",numrow)
    		print("figure number:",i+1)
    		print("num of subplots:",(mxrg-1))
    		if (i == (numfig-1)):
    			print("last figure")
    		totfig = numrow*4		 
    		min_site = situq[(i*24)]
    		max_site = situq[(i*24)+(mxrg-2)]

    		#print(min_site,max_site)   			
    		for j in range (1,mxrg): # 24 subplots per figure unless last figure
    			j0 = j-1
    			jsit = (i*24) + j0
    			#print(jsit)
    			jsite = situq[jsit]
    			#print(jsite) 
    			dframe_site = dframe_master[dframe_master['Site ID'] == jsite]
    			min_depth = min(dframe_site['Num of Depths'])
    			mean_depth = np.around(np.mean(dframe_site['Num of Depths']),decimals=2)
    			dat_sit = dframe_site['Date'].values
    			s_date = dat_sit[0]
    			e_date = dat_sit[len(dat_sit)-1]
    			stemp_sit = dframe_site['Soil Temp'].values
    			lat_sit = np.round(dframe_site['Lat'].iloc[1],decimals=2)
    			lon_sit = np.round(dframe_site['Long'].iloc[1],decimals=2)
    			
    			if (i < (numfig-1)):
    				ax = plt.subplot(6,4,j)
    			elif (i == (numfig-1)):
    				ax = plt.subplot(numrow,4,j)
    			ax.plot(dat_sit,stemp_sit,marker='o',markerfacecolor='dodgerblue',markersize=2,color='royalblue')
    			ax.xaxis.set_major_locator(mdates.YearLocator(5)) #major tick every 5 years
    			ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
    			ax.xaxis.set_minor_locator(mdates.YearLocator(1)) #minor tick every year   					
    			ax.yaxis.set_major_locator(MultipleLocator(10)) #every 10 degrees will be a major tick
    			ax.yaxis.set_minor_locator(MultipleLocator(2)) #every 2 degrees will be a minor tick
    			ax.set_xlim(xmin,xmax)
    			ax.set_ylim(ymin,ymax)
    			handles = [mpl_patches.Rectangle((0,0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] # create dummy, invisible box to include in text legend
    			axtext = []
    			axtext.append('Site: '+str(jsite)+', Lat: '+str(lat_sit)+'$^\circ$N, Lon :'+str(lon_sit)+'$^\circ$, Avg # of Layers Incl: '+str(mean_depth)) #text to include in legend
    			ax.legend(handles, axtext, loc='best', fontsize = 'small', fancybox=False, framealpha=0, handlelength=0, handletextpad=0) #create text as legend
#    			for ax in fig.get_axes():
#    				ax.label_outer()
#    				plt.xticks(rotation=90)

    		if (i == (numfig-1)):
    			for k in range(mxrg,totfig+1):
    				plt.subplot(numrow,4,k).set_visible(False)    			

    		fig.add_subplot(111, frameon=False) #create large subplot which will include the plot labels for plots
    		plt.tick_params(labelcolor='none',bottom=False,left=False) #hide ticks
    		plt.xlabel('Date',fontweight='bold')
    		plt.ylabel('Soil Temp ($^\circ$ C)',fontweight='bold')

    		if (i < (numfig-1)):    			
    			plt.tight_layout()
			
    		pltfil = "".join(["/mnt/data/users/herringtont/soil_temp/plots/layer_average/"+olri+"/"+lyrj+"/"+str(olri)+"_"+lyrj+"_site_"+str(min_site)+"_site_"+str(max_site)+".png"])
    		print(pltfil)
    		plt.savefig(pltfil)
    		plt.close()



    	  
