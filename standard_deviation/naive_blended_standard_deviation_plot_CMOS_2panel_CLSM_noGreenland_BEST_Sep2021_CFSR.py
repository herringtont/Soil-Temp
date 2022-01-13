import os
import glob
import netCDF4
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
import pathlib
import re
import cdo
import skill_metrics as sm
from cdo import Cdo
from netCDF4 import Dataset,num2date # http://unidata.github.io/netcdf4-python/
from natsort import natsorted
from natsort import os_sorted
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
import seaborn as sn
from calendar import isleap
from dateutil.relativedelta import *
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from dateutil.relativedelta import *
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)

############# Set Directories ############

naive_type = ['simple_average']
olr = ['zscore']#['outliers','zscore','IQR']
lyr = ['top_30cm','30cm_300cm']
thr = ['100']#['0','25','50','75','100']
rmp_type = ['con']#['nn','bil','con']

############# Grab Data ##############

for i in rmp_type:
    rmp_type_i = i
    remap_type = ''.join(['remap'+rmp_type_i])

    print("Remap Style:",remap_type)    
    for j in naive_type:
    	naive_type_j = j

    	for k in olr:
    		olr_k = k

    		for l in lyr:
    			lyr_l = l

    			for m in thr:
    				thr_m = m

    				scatter_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CFSR_res/'+str(remap_type)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021_airtemp_CFSR.csv'])
    				dframe_scatter = pd.read_csv(scatter_fil)
    				station_scatter = dframe_scatter['Station'].values
    				naive_all_scatter = dframe_scatter['Ensemble Mean'].values
    				CFSR_scatter = dframe_scatter['CFSR'].values
    				ERA5_scatter = dframe_scatter['ERA5'].values
    				ERA5_Land_scatter = dframe_scatter['ERA5-Land'].values
    				GLDAS_scatter = dframe_scatter['GLDAS-Noah'].values

    				print("Soil Depth:",lyr_l)
    				print("Max Station Temp:",max(station_scatter))
    				print("Min Station Temp:",min(station_scatter))


############# Loop through temperature bins #############


    				if (lyr_l == "top_30cm"):
    					temp_bins = np.arange(-40,37,4)
    				elif (lyr_l == "30cm_300cm"):
    					temp_bins = np.arange(-32,35,4)

    				print(len(temp_bins))
    				len_temp_bins = len(temp_bins)
    				start_temp_master = []
    				end_temp_master = []
    				central_temp_master = []
    				sample_size_master = []
    				station_stdev_master = []
    				naive_stdev_master = []
    				naive_noJRA_stdev_master = []
    				naive_noJRAold_stdev_master = []
    				naive_all_stdev_master = []
    				CFSR_stdev_master = []
    				ERAI_stdev_master = []
    				ERA5_stdev_master = []
    				ERA5_Land_stdev_master = []
    				JRA_stdev_master = []
    				MERRA2_stdev_master = []
    				GLDAS_stdev_master = []
    				GLDAS_CLSM_stdev_master = []
    				for n in range(0,len_temp_bins-1): ##loop through temps
    					start_temp = temp_bins[n]
    					end_temp = temp_bins[n+1]
    					#print("start temp:",start_temp)
    					#print("end temp:",end_temp)

    					temp_range = np.where((station_scatter >= start_temp) & (station_scatter < end_temp)) #get indices of values where station temp is within target bin
    					temp_idx = temp_range[0]
    					sample_size = len(temp_idx)
    					sample_size_master.append(sample_size)
    					central_temp = (start_temp + end_temp)/2
    					start_temp_master.append(start_temp)
    					end_temp_master.append(end_temp)
    					central_temp_master.append(central_temp)

############ create temperature arrays ################

    					station_temp_master = []
    					naive_temp_master = []
    					naive_noJRA_temp_master = []
    					naive_noJRAold_temp_master = []
    					naive_all_temp_master = []
    					CFSR_temp_master = []
    					ERAI_temp_master = []
    					ERA5_temp_master = []
    					ERA5_Land_temp_master = []
    					JRA_temp_master = []
    					MERRA2_temp_master = []
    					GLDAS_temp_master = []
    					GLDAS_CLSM_temp_master = []    					
    					for o in temp_idx:
    						station_stemp = station_scatter[o]
    						station_temp_master.append(station_stemp)
    						naive_all_stemp = naive_all_scatter[o]
    						naive_all_temp_master.append(naive_all_stemp)
    						CFSR_stemp = CFSR_scatter[o]
    						CFSR_temp_master.append(CFSR_stemp)
    						ERA5_stemp = ERA5_scatter[o]
    						ERA5_temp_master.append(ERA5_stemp)
    						ERA5_Land_stemp = ERA5_Land_scatter[o]
    						ERA5_Land_temp_master.append(ERA5_Land_stemp)
    						GLDAS_stemp = GLDAS_scatter[o]
    						GLDAS_temp_master.append(GLDAS_stemp)


    					if (len(station_temp_master) < 10):
    						station_std = np.nan
    					elif (len(station_temp_master) >= 10):
    						station_std = np.std(station_temp_master)
    					station_stdev_master.append(station_std)

    					if (len(naive_all_temp_master) < 10):
    						naive_all_std = np.nan
    					elif (len(naive_all_temp_master) >= 10):
    						naive_all_std = np.std(naive_all_temp_master)
    					naive_all_stdev_master.append(naive_all_std)

    					if (len(CFSR_temp_master) < 10):
    						CFSR_std = np.nan
    					elif (len(CFSR_temp_master) >= 10):
    						CFSR_std = np.std(CFSR_temp_master)
    					CFSR_stdev_master.append(CFSR_std)

    					if (len(ERA5_temp_master) < 10):
    						ERA5_std = np.nan
    					elif (len(ERA5_temp_master) >= 10):
    						ERA5_std = np.std(ERA5_temp_master)
    					ERA5_stdev_master.append(ERA5_std)

    					if (len(ERA5_Land_temp_master) < 10):
    						ERA5_Land_std = np.nan
    					elif (len(ERA5_Land_temp_master) >= 10):
    						ERA5_Land_std = np.std(ERA5_Land_temp_master)
    					ERA5_Land_stdev_master.append(ERA5_Land_std)

    					if (len(GLDAS_temp_master) < 10):
    						GLDAS_std = np.nan
    					elif (len(GLDAS_temp_master) >= 10):
    						GLDAS_std = np.std(GLDAS_temp_master)
    					GLDAS_stdev_master.append(GLDAS_std) 


    				dataframe_stdev = pd.DataFrame(data=start_temp_master,columns=['Start Temp'])
    				dataframe_stdev['Central Temp'] = central_temp_master
    				dataframe_stdev['End Temp'] = end_temp_master
    				dataframe_stdev['Sample Size'] = sample_size_master
    				dataframe_stdev['Station STDEV'] = station_stdev_master
    				dataframe_stdev['Ensemble Mean STDEV'] = naive_all_stdev_master
    				dataframe_stdev['CFSR STDEV'] = CFSR_stdev_master
    				dataframe_stdev['ERA5 STDEV'] = ERA5_stdev_master
    				dataframe_stdev['ERA5-Land STDEV'] = ERA5_Land_stdev_master
    				dataframe_stdev['GLDAS-Noah STDEV'] = GLDAS_stdev_master

    				stdev_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_stdev_temp/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_'+str(naive_type_j)+'_'+str(olr_k)+'_'+str(lyr_l)+'_thr_'+str(thr_m)+'_stdev_temp_4_degree_bins_CMOS_CLSM_noGreenland_BEST_Sep2021_CFSR.csv'])
    				dataframe_stdev.to_csv(stdev_fil,index=False)
    				#print(dataframe_stdev)


    				dframe_sdev = pd.DataFrame(data=start_temp_master,columns=['Start Temp'])
    				#dframe_sdev['Central Temp'] = central_temp_master
    				dframe_sdev['End Temp'] = end_temp_master
    				#dframe_sdev['Sample Size'] = sample_size_master
    				dframe_sdev['Ensemble Mean STDEV'] = naive_all_stdev_master

    				print(dataframe_stdev)
############## Create Figure ##############
    fig = plt.figure()
    fig,axs=plt.subplots(nrows=3,ncols=1,sharex='all',figsize=(20,20))

    matplotlib.rc('xtick', labelsize=25) 
    matplotlib.rc('ytick', labelsize=25)

    top_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_stdev_temp/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_simple_average_zscore_top_30cm_thr_100_stdev_temp_4_degree_bins_CMOS_CLSM_noGreenland_BEST_Sep2021_CFSR.csv'])   				
    btm_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_stdev_temp/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_simple_average_zscore_30cm_300cm_thr_100_stdev_temp_4_degree_bins_CMOS_CLSM_noGreenland_BEST_Sep2021_CFSR.csv'])

    #x_names=['-32($^\circ$ C) to -30($^\circ$ C)','-30($^\circ$ C) to -28($^\circ$ C)','-28($^\circ$ C) to -26($^\circ$ C)','-26($^\circ$ C) to -24($^\circ$ C)','-24($^\circ$ C) to -22($^\circ$ C)','-22($^\circ$ C) to -20($^\circ$ C)','-20($^\circ$ C) to -18($^\circ$ C)','-18($^\circ$ C) to -16($^\circ$ C)','-16($^\circ$ C) to -14($^\circ$ C)','-14($^\circ$ C) to -12($^\circ$ C)','-12($^\circ$ C) to -10($^\circ$ C)','-10($^\circ$ C) to -8($^\circ$ C)','-8($^\circ$ C) to -6($^\circ$ C)','-6($^\circ$ C) to -4($^\circ$ C)','-4($^\circ$ C) to -2($^\circ$ C)','-2($^\circ$ C) to -0($^\circ$ C)','0($^\circ$ C) to 2($^\circ$ C)','2($^\circ$ C) to 4($^\circ$ C)','4($^\circ$ C) to 6($^\circ$ C)','6($^\circ$ C) to 8($^\circ$ C)','8($^\circ$ C) to 10($^\circ$ C)','10($^\circ$ C) to 12($^\circ$ C)','12($^\circ$ C) to 14($^\circ$ C)','14($^\circ$ C) to 16($^\circ$ C)','16($^\circ$ C) to 18($^\circ$ C)','18($^\circ$ C) to 20($^\circ$ C)','20($^\circ$ C) to 22($^\circ$ C)']

    ymin = 0
    ymax = 13

    y_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    top_dframe = pd.read_csv(top_fil)
    top_temp = top_dframe['Central Temp'].values
    top_station_stdev = top_dframe['Station STDEV'].values
    top_naive_all_stdev = top_dframe['Ensemble Mean STDEV'].values
    top_CFSR_stdev = top_dframe['CFSR STDEV'].values 
    top_ERA5_stdev = top_dframe['ERA5 STDEV'].values
    top_ERA5_Land_stdev = top_dframe['ERA5-Land STDEV'].values
    top_GLDAS_stdev = top_dframe['GLDAS-Noah STDEV'].values   

    #print(top_temp)

    ax1 = plt.subplot(2,1,1)
    #ax1.plot(top_temp,top_station_stdev,label='Station',marker='o',markerfacecolor='dimgrey',markersize=2.5,color='dimgrey',linewidth=2.75)
    #ax1.plot(top_temp,top_naive_stdev,label='Naive Blend A',marker='p',markerfacecolor='dodgerblue',markersize=3.0,color='dodgerblue',linewidth=3.0)
    #ax1.plot(top_temp,top_naive_noJRAold_stdev,label='Naive Blend B',marker='p',markerfacecolor='mediumslateblue',markersize=3.0,color='mediumslateblue',linewidth=3.0)
    ax1.plot(top_temp,top_naive_all_stdev,label='Ensemble Mean',marker='p',markerfacecolor='hotpink',markersize=8.0,color='hotpink',linewidth=8.0)
    #ax1.plot(top_temp,top_naive_noJRA_stdev,label='Naive Blend D',marker='p',markerfacecolor='coral',markersize=3.0,color='coral',linewidth=3.0)
    ax1.plot(top_temp,top_CFSR_stdev,label='CFSR',marker='s',markerfacecolor='darkviolet',markersize=3.0,color='darkviolet',linewidth=3.0)
    #ax1.plot(top_temp,top_ERAI_stdev,label="ERA-Interim",marker='v',markerfacecolor='springgreen',markersize=3.0,color='springgreen',linewidth=3.0,linestyle='dotted')
    ax1.plot(top_temp,top_ERA5_stdev,label="ERA5",marker='^',markerfacecolor='greenyellow',markersize=3.0,color='greenyellow',linewidth=3.0,linestyle='dashed')
    ax1.plot(top_temp,top_ERA5_Land_stdev,label="ERA5-Land",marker='^',markerfacecolor='dodgerblue',markersize=3.0,color='dodgerblue',linewidth=3.0) 
    #ax1.plot(top_temp,top_JRA_stdev,label="JRA-55",marker='*',markerfacecolor='red',markersize=3.0,color='red',linewidth=3.0)
    #ax1.plot(top_temp,top_MERRA2_stdev,label="MERRA2",marker='D',markerfacecolor='goldenrod',markersize=3.0,color='goldenrod',linewidth=3.0)
    ax1.plot(top_temp,top_GLDAS_stdev,label="GLDAS-Noah",marker='x',markerfacecolor='black',markersize=3.0,color='black',linewidth=3.0)
    #ax1.plot(top_temp,top_GLDAS_CLSM_stdev,label="GLDAS-CLSM",marker='x',markerfacecolor='darkgrey',markersize=3.0,color='darkgrey',linewidth=3.0)

    handles1 = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
    ax1text = []
    ax1text.append('a) 0cm - 30cm Soil Layer')

    ax1.tick_params(axis="x", labelsize=22)
    ax1.tick_params(axis="y", labelsize=22)

    ax1.set_xticks(top_temp)
    ax1.set_yticks(y_names)
    #ax1.yaxis.set_major_locator(MultipleLocator(1))
    #ax1.yaxis.set_minor_locator(MultipleLocator(0.5))
    
    ax1.set_xlim(-36,32)
    ax1.set_ylim(ymin,ymax)
    #ax1.legend(handles1, ax1text, loc='best', fontsize=18, fancybox=False, framealpha=0, handlelength=0, handletextpad=0)

    btm_dframe = pd.read_csv(btm_fil)
    btm_temp = btm_dframe['Central Temp'].values
    btm_station_stdev = btm_dframe['Station STDEV'].values
    btm_naive_all_stdev = btm_dframe['Ensemble Mean STDEV'].values
    btm_CFSR_stdev = btm_dframe['CFSR STDEV'].values
    btm_ERA5_stdev = btm_dframe['ERA5 STDEV'].values
    btm_ERA5_Land_stdev = btm_dframe['ERA5-Land STDEV'].values
    btm_GLDAS_stdev = btm_dframe['GLDAS-Noah STDEV'].values    

    #print(btm_temp)

    ax3 = plt.subplot(2,1,2)
    #ax3.plot(btm_temp,btm_station_stdev,label='Station',marker='o',markerfacecolor='dimgrey',markersize=2.5,color='dimgrey',linewidth=2.75)
    #ax3.plot(btm_temp,btm_naive_stdev,label='Naive Blend A',marker='p',markerfacecolor='dodgerblue',markersize=3.0,color='dodgerblue',linewidth=3.0)
    #ax3.plot(btm_temp,btm_naive_noJRAold_stdev,label='Naive Blend B',marker='p',markerfacecolor='mediumslateblue',markersize=3.0,color='mediumslateblue',linewidth=3.0)
    ax3.plot(btm_temp,btm_naive_all_stdev,label='Ensemble Mean',marker='p',markerfacecolor='hotpink',markersize=8.0,color='hotpink',linewidth=8.0)
    #ax3.plot(btm_temp,btm_naive_noJRA_stdev,label='Naive Blend D',marker='p',markerfacecolor='coral',markersize=3.0,color='coral',linewidth=3.0)
    ax3.plot(btm_temp,btm_CFSR_stdev,label='CFSR',marker='s',markerfacecolor='darkviolet',markersize=3.0,color='darkviolet',linewidth=3.0)
    #ax3.plot(btm_temp,btm_ERAI_stdev,label="ERA-Interim",marker='v',markerfacecolor='springgreen',markersize=3.0,color='springgreen',linewidth=3.0,linestyle='dotted')
    ax3.plot(btm_temp,btm_ERA5_stdev,label="ERA5",marker='^',markerfacecolor='greenyellow',markersize=3.0,color='greenyellow',linewidth=3.0,linestyle='dashed')
    ax3.plot(btm_temp,btm_ERA5_Land_stdev,label="ERA5-Land",marker='^',markerfacecolor='dodgerblue',markersize=3.0,color='dodgerblue',linewidth=3.0) 
    #ax3.plot(btm_temp,btm_JRA_stdev,label="JRA-55",marker='*',markerfacecolor='red',markersize=3.0,color='red',linewidth=3.0)
    #ax3.plot(btm_temp,btm_MERRA2_stdev,label="MERRA2",marker='D',markerfacecolor='goldenrod',markersize=3.0,color='goldenrod',linewidth=3.0)
    ax3.plot(btm_temp,btm_GLDAS_stdev,label="GLDAS-Noah",marker='x',markerfacecolor='black',markersize=3.0,color='black',linewidth=3.0)
    #ax3.plot(btm_temp,btm_GLDAS_CLSM_stdev,label="GLDAS-CLSM",marker='x',markerfacecolor='darkgrey',markersize=3.0,color='darkgrey',linewidth=3.0)

    handles3 = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
    ax3text = []
    ax3text.append('b) 100cm - 300cm Soil Layer')

    ax3.tick_params(axis="x", labelsize=22)
    ax3.tick_params(axis="y", labelsize=22)

    ax3.set_xticks(top_temp)
    ax3.set_yticks(y_names)
    
    #ax3.yaxis.set_major_locator(MultipleLocator(1))
    #ax3.yaxis.set_minor_locator(MultipleLocator(0.5))
 
    ax3.set_xlim(-36,32)
    ax3.set_ylim(ymin,ymax)
    #ax3.legend(handles3, ax3text, loc='best', fontsize=18, fancybox=False, framealpha=0, handlelength=0, handletextpad=0)
    #ax3.set_xticklabels(x_names)

    lines = []
    labels = []
    for ax in fig.get_axes():
    	axLine, axLabel = ax.get_legend_handles_labels()
    	lines.extend(axLine)
    	labels.extend(axLabel)


    fig.add_subplot(111,frameon=False)
    plt.tick_params(labelcolor='none',bottom=False,left=False)
    plt.xlabel('Station Soil Temperature($^\circ$ C)', fontweight='bold',fontsize=25)
    plt.ylabel('Reanalysis Soil Temp Std Dev($^\circ$ C)',fontweight='bold', fontsize=25)   
    legend = fig.legend(lines[0:5],labels[0:5],loc="upper right",fontsize=25,title="Legend")
    legend.get_title().set_fontsize('25') #legend 'Title' fontsize

    fig_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/naive_blend_stdev_plots/new_data/CFSR_res/'+str(remap_type)+'_all_layers_thr_100_stdev_temp_4_degree_bins_noGreenland_GHCN_Sep2021_CFSR.tiff'])
    plt.tight_layout()
    plt.savefig(fig_fil)
    plt.close()



