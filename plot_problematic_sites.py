import os
import csv
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.patches as mpl_patches
import numpy as np
import scipy
import pandas as pd
import xarray as xr
import seaborn as sns
import pytesmo
import math
import cftime
import re
from decimal import *
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)

###################### set directories ###############################


multi_dir_anom = "/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/anom/multiple_sites/"
temp_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/layer_average/no_outliers/"
grid_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/subset_multiple/"

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']

problem_grids = ['73980','76528','76791']


################# loop through in-situ files ###############
for h in rmp_type: #loops through remap type
    rmph = h
    if(rmph == "nn"):
    	remap_type = "remapnn"
    elif(rmph == "bil"):
    	remap_type = "remapbil"   	 
    for i in olr: #loops throuh outlier type
    	olri = i
    	for j in lyr: #loops through layer
    		lyrj = j
    		for k in thr: #loops through missing threshold
    			thrk = k
    			multi_fil = ''.join([multi_dir_anom+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr_'+str(thrk)+'_multiple_sites_anom.csv'])
    			dframe_multi = pd.read_csv(multi_fil)
    			gcells = dframe_multi['Grid Cell'].values
    			#print(gcells)

################ set up figure specifications #############
    			fig = plt.figure() 					    				
    			fig,axs = plt.subplots(nrows = 1, ncols = 3, figsize=(20,20))
    			a = 0

################# loop through grid cells ####################
    			for l in problem_grids:
    				b = a+1
    				ax = plt.subplot(1,3,b)
    				gcell_l = l
    				#print(gcell_l)
    				grid_fil = ''.join([grid_dir+str(remap_type)+'/'+str(olri)+'/'+str(lyrj)+'/thr_'+str(thrk)+'/grid_'+str(gcell_l)+'_anom.csv'])
    				dframe_grid = pd.read_csv(grid_fil)
    				dates = dframe_grid['Date'].values
    				x = dates
    				#print(dframe_grid)    				
    				col_nam = dframe_grid.columns
    				sit_num = [s for s in col_nam if s.isdigit()]
    				num_sites = len(sit_num)
    				dframe_temp = dframe_grid.drop(columns=['Grid Cell','Central Lat','Central Lon','Spatial Avg Anom','Sites Incl'])
    				print(dframe_temp)
################ problem sites #######################
    				if (gcell_l == '73980'):
    					problem_sites = ['6']
    				elif (gcell_l == '76528'):
    					problem_sites = ['7','140']
    				elif (gcell_l == '76791'):
    					problem_sites = ['34']


################ loop through site numbers ################
    				for m in problem_sites:

    					sitid = m
    					site_temp = dframe_temp[str(sitid)].values
    					#print(site_temp)
					
################ create plots ##################

    					y = site_temp
    					xmin = datetime.date(1990,1,1)
    					xmax = datetime.date(2020,1,1)
    					ymin = -20
    					ymax = 20
    					y = site_temp
    					clrs = ['blue','green','brown','gold', 'lavender','magenta','silver','yellowgreen','goldenrod','salmon','black','cyan']
    					if sitid in problem_sites:
    						wght = 2    						
    						clr_i = ['red']
    						mkr = 's'
    					else:
    						wght = 1
    						clr_i = clrs[a]
    						mkr = 'o'

    					#ax.set_ylim(ymin,ymax)
    					#ax.set_xlim(xmin,xmax)
    					ax.plot(x,y,label='site:'+str(sitid),linewidth=wght,markersize=wght)     					


    				ax.xaxis.set_major_locator(mdates.YearLocator(5)) #major tick every 5 years
    				ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
    				ax.xaxis.set_minor_locator(mdates.YearLocator(1)) #minor tick every year   					
    				ax.yaxis.set_major_locator(MultipleLocator(5)) #every 5 degrees will be a major tick
    				ax.yaxis.set_minor_locator(MultipleLocator(1)) #every 1 degrees will be a minor tick
    				ax.legend(shadow=True, fancybox=True, loc='best')
    				#ax.title('Grid Cell: '+str(gcell_l))

    			a = a+1

    			fig.add_subplot(111, frameon=False) #create large subplot which will include the plot labels for plots
    			plt.tick_params(labelcolor='none',bottom=False,left=False) #hide ticks
    			plt.xlabel('Date',fontweight='bold')
    			plt.ylabel('Soil Temp Anomaly($^\circ$ C)',fontweight='bold')
    			pltfil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/problem_sites/'+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_problem_sites_anom.png'])
    			plt.savefig(pltfil)
    			plt.close()
