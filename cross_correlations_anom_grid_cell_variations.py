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
import xarray as xr
import seaborn as sns
import pytesmo
import math
import cftime
import re
import collections
import pathlib
import seaborn as sn
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from decimal import *
from calendar import isleap
from dateutil.relativedelta import *
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpl_patches

######################## define functions ###################################
def str_to_datetime(column, date_fmt):

    date_list = []

    for dt_str in column:
        new_dt = datetime.datetime.strptime(dt_str, date_fmt)
        date_list.append(new_dt)
			

    return date_list


def load_pandas(file_name):
    print("Loading in-situ file: ", file_name)
    dframe = pd.read_csv(file_name)
    


############################## grab sites and grid cells for each soil layer ##############################
geom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/spatial_join/"
multi_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/multiple_sites/"
multi_dir_anom = "/mnt/data/users/herringtont/soil_temp/In-Situ/grid_cell_subsets/anom/multiple_sites/"
raw_temp_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/subset_multiple/"
anom_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average_anom/subset_multiple/"
olr = ['outliers','zscore','IQR']
lyr = ['0_9.9','10_29.9','30_99.9','100_299.9','300_deeper']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']

sit_master = []
grid_master = []
olr_master = []
lyr_master = []
thr_master = []
lat_master = []
lon_master = []
rmp_master = []

############ loop through remap style ##############
for h in rmp_type:
    rmph = h
    remap_type = ''.join(["remap"+rmph])
############ loop through outlier type #############
    for i in olr:
    	olri = i

############ loop through soil layer ##############    
    	for j in lyr:
    		lyrj = j

############# loop through missing threshold ##########    	
    		for k in thr:
    			thrk = k
    			multi_fil = ''.join([multi_dir+str(remap_type)+"_"+str(olri)+"_"+str(lyrj)+"_thr_"+str(thrk)+"_multiple_sites.csv"]) ##### for raw temperature data
    			multi_fil_anom = ''.join([multi_dir+str(remap_type)+"_"+str(olri)+"_"+str(lyrj)+"_thr_"+str(thrk)+"_multiple_sites.csv"]) ##### for anomaly data		
    			dframe_multi_site = pd.read_csv(multi_fil)
    			gcell = dframe_multi_site['Grid Cell'].values
    			num_of_gcell = len(gcell)
    			#gcell = ['76791']
    			print(gcell)



############# set up correlation matrix figure ###################
    			fig = plt. figure()
    			if (num_of_gcell%4 == 0):
    				num_rows = int(num_of_gcell/4)
    				last_fig = int(num_rows*4)
    			elif (num_of_gcell%3 != 0):
    				num_rows = int((num_of_gcell//4)+1)
    				last_fig = int(num_of_gcell%(num_rows*4))

    			#print("Num of Grid Cells:",num_of_gcell)
    			#print("Num of Rows:",num_rows)
    			#print("Last Figure:",last_fig)
    			fig, axs = plt.subplots(nrows = num_rows, ncols = 4, figsize=(20,20))				   			    								

############# loop through grid cells #################
    			x = 1
    			for l in gcell:
    				gcell_l = l
    				anom_fil = ''.join([anom_dir+str(remap_type)+"/"+str(olri)+"/"+str(lyrj)+"/thr_"+str(thrk)+"/grid_"+str(gcell_l)+"_anom.csv"])
    				print(gcell_l)
    				dframe_anom = pd.read_csv(anom_fil)
    				col_nam = dframe_anom.columns
    				sit_num = [s for s in col_nam if s.isdigit()] ######## check which elements of list are digits (as these are the site numbers)
    				print(sit_num)
    				num_sites = dframe_anom['Sites Incl']
    				tot_sites = len(sit_num)
    				avg_sites = np.mean(num_sites)
    				max_sites = max(num_sites)
    				#print("Max Number of Sites Incl:",max_sites)
    				#print("Avg Number of Sites Incl:",avg_sites) 
    				dframe_anom_only = dframe_anom.drop(columns=['Grid Cell','Central Lat', 'Central Lon', 'Spatial Avg Anom'])
    				if (max_sites < len(sit_num)):
    					dframe_anom_only = dframe_anom_only.loc[(dframe_anom_only['Sites Incl'] == max_sites)]
    					dframe_anom_only = dframe_anom_only.dropna(axis='columns')
    					col_nam = dframe_anom_only.columns
    					sit_num = [s for s in col_nam if s.isdigit()]
    					sites_incl = len(sit_num)
    					#dframe_anom_only = dframe_anom_only.drop(columns=['Sites Incl'])
    					dframe_anom_only['Sites Incl'] = len(sit_num)					
    				#print(dframe_anom_only)
    				else:
    					sites_incl = len(sit_num)
    					dframe_anom_only = dframe_anom_only.dropna()
    				len_dframe = len(dframe_anom)
    				len_anoms = len(dframe_anom_only)
    				pct_incl = (len_anoms/len_dframe)*100
    				#print(dframe_anom_only)
    				dat_mon = dframe_anom['Date'].tolist()
    				date_mon = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dat_mon]
    				date_mon_CFSR = []
    				date_mon_CFSR2 = []				
				

    				lat_cen = round(dframe_anom['Central Lat'].iloc[0],2)
    				lon_cen = round(dframe_anom['Central Lon'].iloc[0],2)


    				dframe_anom_only = dframe_anom_only.drop(columns=['Date','Sites Incl'])
    				#print("Total Number of Rows:",len_dframe)
    				#print("Number of Rows Included:",len_anoms)
    				#print("Percent of Rows Included;",pct_incl)
    				#print(dframe_anom_only)

    				
############## calculate correlation matrix ################
    				ax = plt.subplot(num_rows,4,x)
    				if(len_anoms > 19): ## only calculate 
    					corrMatrix = dframe_anom_only.corr()   				
    					sn.heatmap(round(corrMatrix,2), annot=True, vmin=-1, vmax=1, cmap='seismic').set_title('Lat: '+ str(lat_cen)+'N, Lon: '+str(lon_cen)+', N = '+str(len_anoms))
    				elif(len_anoms <= 19):
    					handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 2
    					labels = []
    					labels.append('Lat: '+ str(lat_cen)+'N, Lon: '+str(lon_cen))
    					labels.append('Sample size insufficient to produce matrix')
    					ax.legend(handles, labels, loc='best', fontsize='medium', fancybox=False, framealpha=0, handlelength=0, handletextpad=0)
    					
    				x = x+1

    			corr_fil = ''.join(["/mnt/data/users/herringtont/soil_temp/plots/corr_matrix_intra_site/"+str(remap_type)+'_'+str(olri)+'_'+str(lyrj)+'_thr'+str(thrk)+'_corr_intra_site_var.png'])
    			print(corr_fil)
    			path = pathlib.Path(corr_fil)
    			path.parent.mkdir(parents=True, exist_ok=True)
    			plt.savefig(corr_fil)
    			plt.close()
