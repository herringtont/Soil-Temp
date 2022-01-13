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


########## Define Functions ##########

def scatter_subset(x, y, hue, mask, **kws):
    sns.scatterplot(x=x[mask], y=y[mask], hue=hue[mask], **kws)
    
    
############# Set Directories ############

naive_type = 'simple_average'
olr = 'zscore'
lyr_top = 'top_30cm'
lyr_btm = '30cm_300cm'
thr = '100'
remap_type = 'remapcon'
temp_thr = '-2C'
permafrost_type = ['RS_2002_none', 'RS_2002_discontinuous', 'RS_2002_continuous','Brown_1970_none', 'Brown_1970_discontinuous', 'Brown_1970_continuous']

sns.set_context(rc={"axes.labelsize":20}, font_scale=1.0)

############# Grab Data ###############
scatter_fil_top = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type)+'_'+str(olr)+'_'+str(lyr_top)+'_thr_'+str(thr)+'_dframe_scatterplot_ERA5_'+str(temp_thr)+'_CMOS_CLSM_subset_permafrost.csv'])
scatter_fil_btm = ''.join(['/mnt/data/users/herringtont/soil_temp/naive_blend_taylor_metrics/new_data/CLSM_res/subset/'+str(remap_type)+'_'+str(naive_type)+'_'+str(olr)+'_'+str(lyr_btm)+'_thr_'+str(thr)+'_dframe_scatterplot_ERA5_'+str(temp_thr)+'_CMOS_CLSM_subset_permafrost.csv'])

dframe_scatter_top = pd.read_csv(scatter_fil_top)
dframe_scatter_top['Layer'] = '0cm - 30cm'

dframe_scatter_btm = pd.read_csv(scatter_fil_btm)
dframe_scatter_btm['Layer'] = '30cm - 300cm'


for x in permafrost_type:
    permafrost_type_x = x

    if (permafrost_type_x == 'RS_2002_none'):
    	dframe_scatter_top_permafrost = dframe_scatter_top[dframe_scatter_top['RS 2002 Permafrost'] == 'none']
    	dframe_scatter_btm_permafrost =  dframe_scatter_btm[dframe_scatter_btm['RS 2002 Permafrost'] == 'none']

    if (permafrost_type_x == 'RS_2002_discontinuous'):
    	dframe_scatter_top_permafrost = dframe_scatter_top[dframe_scatter_top['RS 2002 Permafrost'] == 'discontinuous']
    	dframe_scatter_btm_permafrost =  dframe_scatter_btm[dframe_scatter_btm['RS 2002 Permafrost'] == 'discontinuous']

    if (permafrost_type_x == 'RS_2002_continuous'):
    	dframe_scatter_top_permafrost = dframe_scatter_top[dframe_scatter_top['RS 2002 Permafrost'] == 'continuous']
    	dframe_scatter_btm_permafrost =  dframe_scatter_btm[dframe_scatter_btm['RS 2002 Permafrost'] == 'continuous']

    if (permafrost_type_x == 'Brown_1970_none'):
    	dframe_scatter_top_permafrost = dframe_scatter_top[dframe_scatter_top['Brown 1970 Permafrost'] == 'none']
    	dframe_scatter_btm_permafrost =  dframe_scatter_btm[dframe_scatter_btm['Brown 1970 Permafrost'] == 'none']

    if (permafrost_type_x == 'Brown_1970_discontinuous'):
    	dframe_scatter_top_permafrost = dframe_scatter_top[dframe_scatter_top['Brown 1970 Permafrost'] == 'discontinuous']
    	dframe_scatter_btm_permafrost =  dframe_scatter_btm[dframe_scatter_btm['Brown 1970 Permafrost'] == 'discontinuous']

    if (permafrost_type_x == 'Brown_1970_continuous'):
    	dframe_scatter_top_permafrost = dframe_scatter_top[dframe_scatter_top['Brown 1970 Permafrost'] == 'continuous']
    	dframe_scatter_btm_permafrost =  dframe_scatter_btm[dframe_scatter_btm['Brown 1970 Permafrost'] == 'continuous']

    station_scatter_top_permafrost = dframe_scatter_top_permafrost['Station'].values
    naive_scatter_top_permafrost = dframe_scatter_top_permafrost['Naive Blend'].values
    naive_noJRA_scatter_top_permafrost = dframe_scatter_top_permafrost['Naive Blend no JRA55'].values
    naive_noJRAold_scatter_top_permafrost = dframe_scatter_top_permafrost['Naive Blend no JRA55 Old'].values
    naive_all_scatter_top_permafrost = dframe_scatter_top_permafrost['Naive Blend All'].values
    CFSR_scatter_top_permafrost = dframe_scatter_top_permafrost['CFSR'].values
    ERAI_scatter_top_permafrost = dframe_scatter_top_permafrost['ERA-Interim'].values
    ERA5_scatter_top_permafrost = dframe_scatter_top_permafrost['ERA5'].values
    ERA5_Land_scatter_top_permafrost = dframe_scatter_top_permafrost['ERA5-Land'].values
    JRA_scatter_top_permafrost = dframe_scatter_top_permafrost['JRA55'].values
    MERRA2_scatter_top_permafrost = dframe_scatter_top_permafrost['MERRA2'].values
    GLDAS_scatter_top_permafrost = dframe_scatter_top_permafrost['GLDAS-Noah'].values
    GLDAS_CLSM_scatter_top_permafrost = dframe_scatter_top_permafrost['GLDAS-CLSM'].values
    season_scatter_top_permafrost = dframe_scatter_top_permafrost['Season'].values
    layer_scatter_top_permafrost =  dframe_scatter_top_permafrost['Layer'].values

    station_scatter_btm_permafrost = dframe_scatter_btm_permafrost['Station'].values
    naive_scatter_btm_permafrost = dframe_scatter_btm_permafrost['Naive Blend'].values
    naive_noJRA_scatter_btm_permafrost = dframe_scatter_btm_permafrost['Naive Blend no JRA55'].values
    naive_noJRAold_scatter_btm_permafrost = dframe_scatter_btm_permafrost['Naive Blend no JRA55 Old'].values
    naive_all_scatter_btm_permafrost = dframe_scatter_btm_permafrost['Naive Blend All'].values
    CFSR_scatter_btm_permafrost = dframe_scatter_btm_permafrost['CFSR'].values
    ERAI_scatter_btm_permafrost = dframe_scatter_btm_permafrost['ERA-Interim'].values
    ERA5_scatter_btm_permafrost = dframe_scatter_btm_permafrost['ERA5'].values
    ERA5_Land_scatter_btm_permafrost = dframe_scatter_btm_permafrost['ERA5-Land'].values
    JRA_scatter_btm_permafrost = dframe_scatter_btm_permafrost['JRA55'].values
    MERRA2_scatter_btm_permafrost = dframe_scatter_btm_permafrost['MERRA2'].values
    GLDAS_scatter_btm_permafrost = dframe_scatter_btm_permafrost['GLDAS-Noah'].values
    GLDAS_CLSM_scatter_btm_permafrost = dframe_scatter_btm_permafrost['GLDAS-CLSM'].values
    season_scatter_btm_permafrost = dframe_scatter_btm_permafrost['Season'].values
    layer_scatter_btm_permafrost =  dframe_scatter_btm_permafrost['Layer'].values

    season_depth_top_permafrost = []
    for i in season_scatter_top_permafrost:
    	seas_i = i
    	if (seas_i == 'Cold'):
    		seas_dep = 'Cold_Top'

    	elif (seas_i == 'Warm'):
    		seas_dep = 'Warm_Top'

    	season_depth_top_permafrost.append(seas_dep)    

    season_depth_btm_permafrost = []
    for i in season_scatter_btm_permafrost:
    	seas_i = i
    	if (seas_i == 'Cold'):
    		seas_dep = 'Cold_Top'

    	elif (seas_i == 'Warm'):
    		seas_dep = 'Warm_Top'

    	season_depth_btm_permafrost.append(seas_dep) 

    station_master = []
    naive_all_master = []
    CFSR_master = []
    ERAI_master = []
    ERA5_master = []
    ERA5_Land_master = []
    JRA_master = []
    MERRA2_master = []
    GLDAS_master = []
    GLDAS_CLSM_master = []
    season_master = []
    layer_master = []
    size_master = []
    season_depth_master = []

    station_master.append(station_scatter_top_permafrost)
    station_master.append(station_scatter_btm_permafrost)
    naive_all_master.append(naive_all_scatter_top_permafrost)
    naive_all_master.append(naive_all_scatter_btm_permafrost)
    CFSR_master.append(CFSR_scatter_top_permafrost)
    CFSR_master.append(CFSR_scatter_btm_permafrost)
    ERAI_master.append(ERAI_scatter_top_permafrost)
    ERAI_master.append(ERAI_scatter_btm_permafrost)
    ERA5_master.append(ERA5_scatter_top_permafrost)
    ERA5_master.append(ERA5_scatter_btm_permafrost)
    ERA5_Land_master.append(ERA5_Land_scatter_top_permafrost)
    ERA5_Land_master.append(ERA5_Land_scatter_btm_permafrost)
    JRA_master.append(JRA_scatter_top_permafrost)
    JRA_master.append(JRA_scatter_btm_permafrost)
    MERRA2_master.append(MERRA2_scatter_top_permafrost)
    MERRA2_master.append(MERRA2_scatter_btm_permafrost)
    GLDAS_master.append(GLDAS_scatter_top_permafrost)
    GLDAS_master.append(GLDAS_scatter_btm_permafrost)
    GLDAS_CLSM_master.append(GLDAS_CLSM_scatter_top_permafrost)
    GLDAS_CLSM_master.append(GLDAS_CLSM_scatter_btm_permafrost)
    season_master.append(season_scatter_top_permafrost)
    season_master.append(season_scatter_btm_permafrost)
    season_depth_master.append(season_depth_top_permafrost)
    season_depth_master.append(season_depth_btm_permafrost)
    layer_master.append(layer_scatter_top_permafrost)
    layer_master.append(layer_scatter_btm_permafrost)

    station_master = [i for sub in station_master for i in sub]
    naive_all_master = [i for sub in naive_all_master for i in sub]
    CFSR_master = [i for sub in CFSR_master for i in sub]
    ERAI_master = [i for sub in ERAI_master for i in sub]
    ERA5_master = [i for sub in ERA5_master for i in sub]
    ERA5_Land_master = [i for sub in ERA5_Land_master for i in sub]
    JRA_master = [i for sub in JRA_master for i in sub]
    MERRA2_master = [i for sub in MERRA2_master for i in sub]
    GLDAS_master = [i for sub in GLDAS_master for i in sub]
    GLDAS_CLSM_master = [i for sub in GLDAS_CLSM_master for i in sub]
    season_master = [i for sub in season_master for i in sub]
    season_depth_master = [i for sub in season_depth_master for i in sub]
    layer_master = [i for sub in layer_master for i in sub]

    dframe_master = pd.DataFrame({'Station':station_master,'Season':season_depth_master,'Layer':layer_master,'Ensemble Mean':naive_all_master,'CFSR':CFSR_master,'ERA-Interim':ERAI_master,'ERA5':ERA5_master,'ERA5-Land':ERA5_Land_master,'JRA55':JRA_master,'MERRA2':MERRA2_master,'GLDAS-Noah':GLDAS_master,'GLDAS-CLSM':GLDAS_CLSM_master})
    total_fil = '/mnt/data/users/herringtont/soil_temp/scripts/dframe_master_'+str(permafrost_type_x)+'.csv'
    dframe_master.to_csv(total_fil,index=False,na_rep=np.nan)

    print(dframe_master)

#################### Create Scatterplot Matrices ####################					

    scatter1 = sn.PairGrid(dframe_master,hue='Season')

    sns.set_context(rc={"axes.labelsize":20}, font_scale=1.0)
    scatter1.map_upper(scatter_subset, mask=dframe_master['Layer'] == '0cm - 30cm', cmap=plt.get_cmap('tab10'))
    scatter1.map_diag(sns.histplot, legend = False, color=plt.get_cmap('tab10'))
    scatter1.map_lower(scatter_subset, mask=dframe_master['Layer'] == '30cm - 300cm',cmap=plt.get_cmap('Dark2'))


    scatter1.set(xlim=(-40,40),ylim=(-40,40))
    plt.savefig('/mnt/data/users/herringtont/soil_temp/plots/naive_blend_scatterplots/new_data/CMOS_poster/CLSM_res/'+str(remap_type)+'_thr_100_scatterplot_all_temp_new_data_CMOS_CLSM_all_layers_'+str(permafrost_type_x)+'.png')
    plt.close()








































