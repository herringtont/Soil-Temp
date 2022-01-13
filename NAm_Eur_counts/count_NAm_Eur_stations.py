import numpy as np
import os
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
lyr = ['top_30cm','30cm_300cm']
wkdir = '/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'


############# Grab Data #################

for i in lyr: # loop through layers
    lyr_i = i

    if (lyr_i == 'top_30cm'):
    	dep = 'top_30cm'

    elif(lyr_i == '30cm_300cm'):
    	dep = '30_299.9'

    print('the layer is:',lyr_i)
    wfil = ''.join([wkdir,'remapcon_'+lyr_i+'_thr_100_dframe_scatterplot_CMOS_CLSM_subset_permafrost_cold_warm_BEST_Sep2021.csv'])
    #print(wfil)
    dframe = pd.read_csv(wfil)
    continent = dframe['Continent']


    NAm_sites_master = []
    Eur_sites_master = []


    dframe_NAm = dframe[dframe['Continent'] == 'North_America']
    gcells_NAm = dframe_NAm['Grid Cell'].values
    gcells_NAm_uq = np.unique(gcells_NAm)

    dframe_Eur = dframe[dframe['Continent'] == 'Eurasia']
    gcells_Eur = dframe_Eur['Grid Cell'].values
    gcells_Eur_uq = np.unique(gcells_Eur)

    for k in gcells_NAm_uq: # loop through grid cells
    	gcell_k = k
    	#print(gcell_k)
    	sptl_avg_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/remapcon/no_outliers/zscore/'+dep+'/thr_100/CLSM/grid_'+str(gcell_k)+'.csv'])
    	file_exists = os.path.exists(sptl_avg_fil)

    	if (file_exists == False):
    		continue
    	dframe_sptl_avg = pd.read_csv(sptl_avg_fil)
    	sites_incl = dframe_sptl_avg['Sites Incl'].values
    	num_sites =  scipy.stats.mode(sites_incl)[0].tolist()

    	NAm_sites_master.append(num_sites)

    for k in gcells_Eur_uq: # loop through grid cells
    	gcell_k = k
    	#print(gcell_k)
    	sptl_avg_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/remapcon/no_outliers/zscore/'+dep+'/thr_100/CLSM/grid_'+str(gcell_k)+'.csv'])

    	file_exists = os.path.exists(sptl_avg_fil)

    	if (file_exists == False):
    		continue
    	dframe_sptl_avg = pd.read_csv(sptl_avg_fil)
    	sites_incl = dframe_sptl_avg['Sites Incl'].values
    	num_sites =  scipy.stats.mode(sites_incl)[0].tolist()
    	Eur_sites_master.append(num_sites)


    NAm_sites_master = [i for sub in NAm_sites_master for i in sub]
    Eur_sites_master = [i for sub in Eur_sites_master for i in sub]

    tot_sites_NAm = np.sum(NAm_sites_master)
    tot_sites_Eur = np.sum(Eur_sites_master)
    print('North America:',tot_sites_NAm)
    print('Eurasia:',tot_sites_Eur)
		
    
