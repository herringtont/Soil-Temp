import os
import glob
import netCDF4
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
import pathlib
import scipy.stats as stats
import cdo
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
from scipy.stats import pearsonr
from dateutil.relativedelta import *
import statistics

######## Files and Directories ##########

top_grid_fil = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/top_30cm_all_zones_summary.csv"
btm_grid_fil = "/mnt/data/users/herringtont/soil_temp/naive_blend_timeseries/summary_fil/new_data/CLSM_res/30cm_300cm_all_zones_summary.csv"

top_soil_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/remapcon/no_outliers/zscore/top_30cm/thr_100/CLSM/Sep2021/"
btm_soil_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/remapcon/no_outliers/zscore/30_299.9/thr_100/CLSM/Sep2021/"


top_summary_fil = '/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/remapcon_top_30cm_thr_100_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021.csv'
btm_summary_fil = '/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/remapcon_30cm_300cm_thr_100_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021.csv'



######## Grab Data ######

dframe_top_grid = pd.read_csv(top_grid_fil)
dframe_btm_grid = pd.read_csv(btm_grid_fil)

dframe_summary_fil_top = pd.read_csv(top_summary_fil)
dframe_summary_fil_btm = pd.read_csv(btm_summary_fil)

top_grids = dframe_top_grid['Grid Cell']
btm_grids = dframe_btm_grid['Grid Cell']

dframe_top_final_cold = "None"
dframe_top_final_warm = "None"





####### Top Level ####

grid_cell_master_top = []
std_dev_master_top = []
letter_master_top = []
mode_sites_incl_top = []
for i in top_grids: #loop through grid cells
    soil_std_master = []
    grid_i = i

    grid_num = str(grid_i)
    #print("Grid Cell:",grid_num)
    soil_fil = ''.join([top_soil_dir+'grid_'+grid_num+'_Sep2021.csv'])
    #print(soil_fil)
    dframe_soil_fil = pd.read_csv(soil_fil)
    sites_incl = dframe_soil_fil['Sites Incl'].values
    avg_sites_incl = np.mean(sites_incl)
    if (avg_sites_incl < 2): #skip grid cells with less than 2 sites included
    	#print("Grid Cell:",grid_num," contains less than 2 sites")
    	continue

    mode_sites_incl = stats.mode(sites_incl)[0].tolist()

    #print(mode_sites_incl)
    
    print('Grid Cell:',grid_num)

    if (i == 2360):
    	letter = 'A'
    elif (i == 2959):
    	letter = 'B'
    elif (i == 3107):
    	letter = 'C'
    elif (i == 3555):
    	letter = 'D'
    elif (i == 4594):
    	letter = 'E'
    elif (i == 6230):
    	letter = 'F'
    elif (i == 7424):
    	letter = 'G'
    elif (i == 16963):
    	letter = 'H'  

    dframe_site_temps = dframe_soil_fil.drop(columns=['Grid Cell','Central Lat','Central Lon','Sites Incl'])

    #print(dframe_site_temps)
    dframe_col_names = dframe_site_temps.drop(columns=['Date','Spatial Avg'])

    dframe_dates = dframe_site_temps['Date'].values

    #print(dframe_dates)
    col_names = dframe_col_names.columns.values   



    std_dev_time = dframe_col_names.std(axis=1,skipna=True).values
    std_dev_time = std_dev_time[~np.isnan(std_dev_time)]
    std_dev_time = std_dev_time[np.where(std_dev_time > 0)]



    dframe_grid_cell_top = pd.DataFrame(data=std_dev_time,columns=['Std Dev'])
    dframe_grid_cell_top['Grid Cell'] = grid_num
    dframe_grid_cell_top['Letter'] = letter
    dframe_grid_cell_top['Mode'] = mode_sites_incl[0]

    #print(dframe_grid_cell_top)

    std_dev_time = dframe_grid_cell_top['Std Dev'].values.tolist()
    std_dev_master_top.append(std_dev_time)
    grid_cell = dframe_grid_cell_top['Grid Cell'].values.tolist()
    grid_cell_master_top.append(grid_cell)
    letter_gcell = dframe_grid_cell_top['Letter'].values.tolist()
    letter_master_top.append(letter_gcell)
    mode_gcell = dframe_grid_cell_top['Mode'].values.tolist()
    mode_sites_incl_top.append(mode_gcell)

grid_cell_master_btm = []
std_dev_master_btm = []
letter_master_btm = []
mode_sites_incl_btm = []
for j in btm_grids: #loop through grid cells
    soil_std_master = []
    grid_i = j

    grid_num = str(grid_i)
    soil_fil = ''.join([btm_soil_dir+'grid_'+grid_num+'_Sep2021.csv'])
    dframe_soil_fil = pd.read_csv(soil_fil)
    sites_incl = dframe_soil_fil['Sites Incl'].values
    avg_sites_incl = np.mean(sites_incl)

    if (avg_sites_incl < 2): #skip grid cells with less than 2 sites included
    	#print("Grid Cell:",grid_num," contains less than 2 sites")
    	continue

    mode_sites_incl = stats.mode(sites_incl)[0].tolist()


    if (i == 2360):
    	letter = 'A'
    elif (i == 2959):
    	letter = 'B'
    elif (i == 3107):
    	letter = 'C'
    elif (i == 3555):
    	letter = 'D'
    elif (i == 4594):
    	letter = 'E'
    elif (i == 6230):
    	letter = 'F'
    elif (i == 7424):
    	letter = 'G'
    elif (i == 16963):
    	letter = 'H'

    print('Grid Cell:',grid_num)

    dframe_site_temps = dframe_soil_fil.drop(columns=['Grid Cell','Central Lat','Central Lon','Sites Incl'])

    #print(dframe_site_temps)
    dframe_col_names = dframe_site_temps.drop(columns=['Date','Spatial Avg'])

    dframe_dates = dframe_site_temps['Date'].values

    #print(dframe_dates)
    col_names = dframe_col_names.columns.values   



    std_dev_time = dframe_col_names.std(axis=1,skipna=True).values
    std_dev_time = std_dev_time[~np.isnan(std_dev_time)]
    std_dev_time = std_dev_time[np.where(std_dev_time > 0)]



    dframe_grid_cell_btm = pd.DataFrame(data=std_dev_time,columns=['Std Dev'])
    dframe_grid_cell_btm['Grid Cell'] = grid_num
    dframe_grid_cell_btm['Letter'] = letter
    dframe_grid_cell_btm['Mode'] = mode_sites_incl[0]

    std_dev_time = dframe_grid_cell_btm['Std Dev'].values.tolist()
    std_dev_master_btm.append(std_dev_time)
    grid_cell = dframe_grid_cell_btm['Grid Cell'].values.tolist()
    grid_cell_master_btm.append(grid_cell)
    letter_gcell = dframe_grid_cell_btm['Letter'].values.tolist()
    letter_master_btm.append(letter_gcell)
    mode_gcell = dframe_grid_cell_btm['Mode'].values.tolist()
    mode_sites_incl_btm.append(mode_gcell)

    
std_dev_master_top = [i for sub in std_dev_master_top for i in sub]
grid_cell_master_top = [i for sub in grid_cell_master_top for i in sub]
letter_master_top = [i for sub in letter_master_top for i in sub]
mode_sites_incl_top = [i for sub in mode_sites_incl_top for i in sub]

dframe_final_top = pd.DataFrame(data=grid_cell_master_top,columns=['Grid Cell'])
dframe_final_top['Letter'] = letter_master_top
dframe_final_top['Soil Temp Std Dev'] = std_dev_master_top
dframe_final_top['Mode Sites'] = mode_sites_incl_top

std_dev_master_btm = [i for sub in std_dev_master_btm for i in sub]
grid_cell_master_btm = [i for sub in grid_cell_master_btm for i in sub]
letter_master_btm = [i for sub in letter_master_btm for i in sub]
mode_sites_incl_btm = [i for sub in mode_sites_incl_btm for i in sub]

dframe_final_btm = pd.DataFrame(data=grid_cell_master_btm,columns=['Grid Cell'])
dframe_final_btm['Letter'] = letter_master_btm
dframe_final_btm['Soil Temp Std Dev'] = std_dev_master_btm
dframe_final_btm['Mode Sites'] = mode_sites_incl_btm


top_letters = dframe_final_top['Letter'].values
top_mode = dframe_final_top['Mode Sites'].values


#print(top_letters)
#print(top_mode)

len_letter = len(dframe_final_top)

letter_mode = []
for i in range(0,len_letter):
    letter_i = top_letters[i]
    mode_i = top_mode[i]    

    mode_letter = ''.join([letter_i+' ('+str(mode_i)+')'])
    letter_mode.append(mode_letter)

dframe_final_top['Mode_Letter'] = letter_mode


print(dframe_final_top)


fig_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/obs_uncertainty/Obs_Uncertainty_Boxplot_toplayer.png'])

sns.set(font_scale = 1.6)
fig = plt.subplots(1,1,figsize=(10,10),sharey=True)
ax = sns.boxplot(x='Mode_Letter', y='Soil Temp Std Dev', data=dframe_final_top)
#ax = sns.boxplot(x='Grid Cell', y='Soil Temp Std Dev', data=dframe_final_top, ax=axs[0])
ax.set_xlabel('Grid Cell (Num of Sites)', size=20, fontweight='bold')
ax.set_ylabel('Standard Deviation ($^\circ$C)', size=20, fontweight='bold')
ax.set_yticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])
ax.set_ylim(0,8)

#sns.boxplot(x='Grid Cell', y='Soil Temp Std Dev', data=dframe_final_btm,ax=axs[1])
#axs[1].set_xlabel('Grid Cell',size = 20)
#axs[1].set_ylabel('Standard Deviation ($^\circ$C)',size=20, fontweight='bold')
#axs[1].set_yticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8])
#axs[1].set_ylim(0,8)

plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')
  
plt.savefig(fig_fil)


