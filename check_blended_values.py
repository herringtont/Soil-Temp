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
import time
import cftime
import pathlib
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
import seaborn as sn
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from decimal import *
from datetime import date

################################## Set Base Directory for Individual Products ######################

prod_dir = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/common_grid/remapnn/common_date/"
stn_dir = "/mnt/data/users/herringtont/soil_temp/In-Situ/All/spatial_average/remapnn/no_outliers/zscore/0_9.9/thr_100/"
################################ Grab data from CSV files ##############################
blended_fil_ncks = ''.join(['/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/TC_blended/raw/remapnn_zscore_0_9.9_thr100_TC_blended_ncks.csv'])
TC_metrics_fil_ncks = ''.join(['/mnt/data/users/herringtont/soil_temp/TC_Analysis/grid_cell_level/raw/remapnn_zscore_0_9.9_thr100_TC_output_gcell_level_raw_ncks.csv'])

blended_fil_cdo = ''.join(['/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/TC_blended/raw/remapnn_zscore_0_9.9_thr100_TC_blended_cdo.csv'])
TC_metrics_fil_cdo = ''.join(['/mnt/data/users/herringtont/soil_temp/TC_Analysis/grid_cell_level/raw/remapnn_zscore_0_9.9_thr100_TC_output_gcell_level_raw_cdo.csv'])



SF_MERRA2_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_MERRA2_SF.nc'])
SF_MERRA2_rnys_fil = xr.open_dataset(SF_MERRA2_rnys_fi)
SF_MERRA2_rnys = SF_MERRA2_rnys_fil['MERRA2_SF']

SF_GLDAS_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_GLDAS_SF.nc'])
SF_GLDAS_rnys_fil = xr.open_dataset(SF_GLDAS_rnys_fi)
SF_GLDAS_rnys = SF_GLDAS_rnys_fil['GLDAS_SF']

EV_JRA_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_JRA55_err_var_cov.nc'])
EV_JRA_rnys_fil = xr.open_dataset(EV_JRA_rnys_fi)
EV_JRA_rnys = EV_JRA_rnys_fil['err_var_JRA55_cov']

EV_MERRA2_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_MERRA2_err_var_cov.nc'])
EV_MERRA2_rnys_fil = xr.open_dataset(EV_MERRA2_rnys_fi)
EV_MERRA2_rnys = EV_MERRA2_rnys_fil['err_var_MERRA2_cov']

EV_GLDAS_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_GLDAS_err_var_cov.nc'])
EV_GLDAS_rnys_fil = xr.open_dataset(EV_GLDAS_rnys_fi)
EV_GLDAS_rnys = EV_GLDAS_rnys_fil['err_var_GLDAS_cov']

SNR_JRA_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_JRA55_SNR.nc'])
SNR_JRA_rnys_fil = xr.open_dataset(SNR_JRA_rnys_fi)
SNR_JRA_rnys = SNR_JRA_rnys_fil['SNR_JRA55']

SNR_MERRA2_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_MERRA2_SNR.nc'])
SNR_MERRA2_rnys_fil = xr.open_dataset(SNR_MERRA2_rnys_fi)
SNR_MERRA2_rnys = SNR_MERRA2_rnys_fil['SNR_MERRA2']

SNR_GLDAS_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_GLDAS_SNR.nc'])
SNR_GLDAS_rnys_fil = xr.open_dataset(SNR_GLDAS_rnys_fi)
SNR_GLDAS_rnys = SNR_GLDAS_rnys_fil['SNR_GLDAS']

fMSE_JRA_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_JRA55_fMSE.nc'])
fMSE_JRA_rnys_fil = xr.open_dataset(fMSE_JRA_rnys_fi)
fMSE_JRA_rnys = fMSE_JRA_rnys_fil['fMSE_JRA55']

fMSE_MERRA2_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_MERRA2_fMSE.nc'])
fMSE_MERRA2_rnys_fil = xr.open_dataset(fMSE_MERRA2_rnys_fi)
fMSE_MERRA2_rnys = fMSE_MERRA2_rnys_fil['fMSE_MERRA2']

fMSE_GLDAS_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/remapnn_GLDAS_fMSE.nc'])
fMSE_GLDAS_rnys_fil = xr.open_dataset(fMSE_GLDAS_rnys_fi)
fMSE_GLDAS_rnys = fMSE_GLDAS_rnys_fil['fMSE_GLDAS']

JRA_wght_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/blended_products/remapnn_JRA55_weights_raw.nc'])
JRA_wght_rnys_fil = xr.open_dataset(JRA_wght_rnys_fi)
JRA_wght_rnys = JRA_wght_rnys_fil['JRA55_weight_raw']

MERRA2_wght_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/blended_products/remapnn_MERRA2_weights_raw.nc'])
MERRA2_wght_rnys_fil = xr.open_dataset(MERRA2_wght_rnys_fi)
MERRA2_wght_rnys = MERRA2_wght_rnys_fil['MERRA2_weight_raw']

GLDAS_wght_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/blended_products/remapnn_GLDAS_weights_raw.nc'])
GLDAS_wght_rnys_fil = xr.open_dataset(GLDAS_wght_rnys_fi)
GLDAS_wght_rnys = GLDAS_wght_rnys_fil['GLDAS_weight_raw']

TC_blended_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/blended_products/remapnn_TC_blended_raw.nc'])
TC_blended_rnys_fil = xr.open_dataset(TC_blended_rnys_fi)
TC_blended_rnys = TC_blended_rnys_fil['TC_blended_stemp']

naive_blended_rnys_fi = ''.join(['/mnt/data/users/herringtont/soil_temp/global_triple_collocation/raw_temp/remapnn/blended_products/remapnn_naive_blended_raw.nc'])
naive_blended_rnys_fil = xr.open_dataset(naive_blended_rnys_fi)
naive_blended_rnys = naive_blended_rnys_fil['naive_blended_stemp']

blended_dframe_ncks = pd.read_csv(blended_fil_ncks)
blended_dframe_cdo = pd.read_csv(blended_fil_cdo)

TC_metrics_dframe_ncks = pd.read_csv(TC_metrics_fil_ncks)
TC_metrics_dframe_cdo = pd.read_csv(TC_metrics_fil_cdo)

gcells = TC_metrics_dframe_ncks['Grid Cell'].values

gcell_master = []
gcell_master_blended = []
lat_master = []
lon_master = []
lat_cen_master = []
lon_cen_master = []
date_master = []

SF_MERRA2_ncks_master = []
SF_GLDAS_ncks_master = []
SNR_JRA_ncks_master = []
EV_JRA_ncks_master = []
EV_JRA_rnys_master = []
EV_MERRA2_ncks_master = []
EV_MERRA2_rnys_master = []
EV_GLDAS_ncks_master = []
EV_GLDAS_rnys_master = []
SNR_MERRA2_ncks_master = []
SNR_GLDAS_ncks_master = []
JRA_wght_ncks_master = []
MERRA2_wght_ncks_master = []
GLDAS_wght_ncks_master = []
JRA_fMSE_ncks_master = []
MERRA2_fMSE_ncks_master = []
GLDAS_fMSE_ncks_master = []
TC_blend_ncks_master = []
naive_blend_ncks_master = []

SF_MERRA2_cdo_master = []
SF_GLDAS_cdo_master = []
SNR_JRA_cdo_master = []
EV_JRA_cdo_master = []
EV_JRA_rnys_master = []
EV_MERRA2_cdo_master = []
EV_MERRA2_rnys_master = []
EV_GLDAS_cdo_master = []
EV_GLDAS_rnys_master = []
SNR_MERRA2_cdo_master = []
SNR_GLDAS_cdo_master = []
JRA_wght_cdo_master = []
MERRA2_wght_cdo_master = []
GLDAS_wght_cdo_master = []
JRA_fMSE_cdo_master = []
MERRA2_fMSE_cdo_master = []
GLDAS_fMSE_cdo_master = []
TC_blend_cdo_master = []
naive_blend_cdo_master = []

JRA_stemp_master = []
MERRA2_stemp_master = []
GLDAS_stemp_master = []
stn_stemp_master = []

SF_MERRA2_rnys_master = []
SF_GLDAS_rnys_master = []
SNR_JRA_rnys_master = []
SNR_MERRA2_rnys_master = []
SNR_GLDAS_rnys_master = []
JRA_wght_rnys_master = []
MERRA2_wght_rnys_master = []
GLDAS_wght_rnys_master = []
JRA_fMSE_rnys_master = []
MERRA2_fMSE_rnys_master = []
GLDAS_fMSE_rnys_master = []
TC_blend_rnys_master = []
naive_blend_rnys_master = []

for i in gcells:
    gcell_i = i
    gcell_master.append(gcell_i)
    blended_dframe_gcell_ncks = blended_dframe_ncks[blended_dframe_ncks['Grid Cell'] == i]
    dates = blended_dframe_gcell_ncks['Date']
    blended_dframe_gcell_ncks['DateTime'] = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in dates]
    blended_dframe_gcell_cdo = blended_dframe_cdo[blended_dframe_cdo['Grid Cell'] == i]
    blended_dframe_gcell_ncks = blended_dframe_gcell_ncks.set_index("DateTime")
    #print(dates)
    TC_metrics_dframe_gcell_ncks = TC_metrics_dframe_ncks[TC_metrics_dframe_ncks['Grid Cell'] == i]
    TC_metrics_dframe_gcell_cdo = TC_metrics_dframe_cdo[TC_metrics_dframe_cdo['Grid Cell'] == i] 
    start = datetime.datetime.strptime("1980-01-01",'%Y-%m-%d')
    end = datetime.datetime.strptime("2018-12-01",'%Y-%m-%d')	
    date_array = [start + relativedelta(months=+x) for x in range(0,468)]
    dattim = [t.strftime('%Y-%m-%d') for t in date_array]
    date_master.append(dattim)    
    lat_cen = blended_dframe_gcell_ncks['Central Lat'].iloc[0]
    lat_master.append(lat_cen)
    lon_cen = blended_dframe_gcell_ncks['Central Lon'].iloc[0]
    lon_master.append(lon_cen)
    lat_cen2 = blended_dframe_gcell_ncks['Central Lat'].values
    lon_cen2 = blended_dframe_gcell_ncks['Central Lon'].values
    gcell_blended = blended_dframe_gcell_ncks['Grid Cell'].values
    if (gcell_i == 105333):
    	blended_dframe_gcell_ncks = blended_dframe_gcell_ncks.reindex(date_array, fill_value=np.nan)
    	blended_dframe_gcell_ncks['Date'] = dattim	
    	blended_dframe_gcell_ncks['Central Lat'] = lat_cen
    	blended_dframe_gcell_ncks['Central Lon'] = lon_cen
    	blended_dframe_gcell_ncks['Grid Cell'] = gcell_i
    	gcell_blended = blended_dframe_gcell_ncks['Grid Cell'].values
    	lat_cen2 = blended_dframe_gcell_ncks['Central Lat'].values
    	lon_cen2 = blended_dframe_gcell_ncks['Grid Cell'].values 			
    gcell_master_blended.append(gcell_blended)
    lat_cen_master.append(lat_cen2)
    lon_cen_master.append(lon_cen2)

    SF_MERRA2_ncks = TC_metrics_dframe_gcell_ncks['Scale_Factor_y_Cov'].iloc[0]
    SF_MERRA2_ncks_master.append(SF_MERRA2_ncks)
    SF_GLDAS_ncks = TC_metrics_dframe_gcell_ncks['Scale_Factor_z_Cov'].iloc[0]
    SF_GLDAS_ncks_master.append(SF_GLDAS_ncks)
    EV_JRA_ncks =  TC_metrics_dframe_gcell_ncks['E_x_Cov'].iloc[0]
    EV_JRA_ncks_master.append(EV_JRA_ncks)
    EV_MERRA2_ncks =  TC_metrics_dframe_gcell_ncks['E_y_Cov'].iloc[0]
    EV_MERRA2_ncks_master.append(EV_MERRA2_ncks)
    EV_GLDAS_ncks =  TC_metrics_dframe_gcell_ncks['E_z_Cov'].iloc[0]
    EV_GLDAS_ncks_master.append(EV_GLDAS_ncks)    
    SNR_JRA_ncks = TC_metrics_dframe_gcell_ncks['SNR_x'].iloc[0]
    SNR_JRA_ncks_master.append(SNR_JRA_ncks)
    SNR_MERRA2_ncks = TC_metrics_dframe_gcell_ncks['SNR_y'].iloc[0]
    SNR_MERRA2_ncks_master.append(SNR_MERRA2_ncks)    
    SNR_GLDAS_ncks = TC_metrics_dframe_gcell_ncks['SNR_z'].iloc[0]
    SNR_GLDAS_ncks_master.append(SNR_GLDAS_ncks)
    JRA_fMSE_ncks = TC_metrics_dframe_gcell_ncks['fMSE_x'].iloc[0]
    JRA_fMSE_ncks_master.append(JRA_fMSE_ncks)
    MERRA2_fMSE_ncks = TC_metrics_dframe_gcell_ncks['fMSE_y'].iloc[0]
    MERRA2_fMSE_ncks_master.append(MERRA2_fMSE_ncks)    
    GLDAS_fMSE_ncks = TC_metrics_dframe_gcell_ncks['fMSE_z'].iloc[0]
    GLDAS_fMSE_ncks_master.append(GLDAS_fMSE_ncks)     
    JRA_wght_ncks = blended_dframe_gcell_ncks['JRA weight'].iloc[0]
    JRA_wght_ncks_master.append(JRA_wght_ncks)
    MERRA2_wght_ncks = blended_dframe_gcell_ncks['MERRA2 weight'].iloc[0]
    MERRA2_wght_ncks_master.append(MERRA2_wght_ncks)
    GLDAS_wght_ncks = blended_dframe_gcell_ncks['GLDAS weight'].iloc[0]
    GLDAS_wght_ncks_master.append(GLDAS_wght_ncks)
    TC_blend_ncks = blended_dframe_gcell_ncks['TC Blended'].values
    TC_blend_ncks_master.append(TC_blend_ncks)
    naive_blend_ncks = blended_dframe_gcell_ncks['Naive Blended'].values
    naive_blend_ncks_master.append(naive_blend_ncks) 

    SF_MERRA2_cdo = TC_metrics_dframe_gcell_cdo['Scale_Factor_y_Cov'].iloc[0]
    SF_MERRA2_cdo_master.append(SF_MERRA2_cdo)
    SF_GLDAS_cdo = TC_metrics_dframe_gcell_cdo['Scale_Factor_z_Cov'].iloc[0]
    SF_GLDAS_cdo_master.append(SF_GLDAS_cdo)
    EV_JRA_cdo =  TC_metrics_dframe_gcell_cdo['E_x_Cov'].iloc[0]
    EV_JRA_cdo_master.append(EV_JRA_cdo)
    EV_MERRA2_cdo =  TC_metrics_dframe_gcell_cdo['E_y_Cov'].iloc[0]
    EV_MERRA2_cdo_master.append(EV_MERRA2_cdo)
    EV_GLDAS_cdo =  TC_metrics_dframe_gcell_cdo['E_z_Cov'].iloc[0]
    EV_GLDAS_cdo_master.append(EV_GLDAS_cdo)    
    SNR_JRA_cdo = TC_metrics_dframe_gcell_cdo['SNR_x'].iloc[0]
    SNR_JRA_cdo_master.append(SNR_JRA_cdo)
    SNR_MERRA2_cdo = TC_metrics_dframe_gcell_cdo['SNR_y'].iloc[0]
    SNR_MERRA2_cdo_master.append(SNR_MERRA2_cdo)    
    SNR_GLDAS_cdo = TC_metrics_dframe_gcell_cdo['SNR_z'].iloc[0]
    SNR_GLDAS_cdo_master.append(SNR_GLDAS_cdo)
    JRA_fMSE_cdo = TC_metrics_dframe_gcell_cdo['fMSE_x'].iloc[0]
    JRA_fMSE_cdo_master.append(JRA_fMSE_cdo)
    MERRA2_fMSE_cdo = TC_metrics_dframe_gcell_cdo['fMSE_y'].iloc[0]
    MERRA2_fMSE_cdo_master.append(MERRA2_fMSE_cdo)    
    GLDAS_fMSE_cdo = TC_metrics_dframe_gcell_cdo['fMSE_z'].iloc[0]
    GLDAS_fMSE_cdo_master.append(GLDAS_fMSE_cdo)     
    JRA_wght_cdo = blended_dframe_gcell_cdo['JRA weight'].iloc[0]
    JRA_wght_cdo_master.append(JRA_wght_cdo)
    MERRA2_wght_cdo = blended_dframe_gcell_cdo['MERRA2 weight'].iloc[0]
    MERRA2_wght_cdo_master.append(MERRA2_wght_cdo)
    GLDAS_wght_cdo = blended_dframe_gcell_cdo['GLDAS weight'].iloc[0]
    GLDAS_wght_cdo_master.append(GLDAS_wght_cdo)
    TC_blend_cdo = blended_dframe_gcell_cdo['TC Blended'].values
    TC_blend_cdo_master.append(TC_blend_cdo)
    naive_blend_cdo = blended_dframe_gcell_cdo['Naive Blended'].values
    naive_blend_cdo_master.append(naive_blend_cdo) 

    stn_fil = ''.join([stn_dir,'grid_',str(gcell_i),'.csv'])
    stn_dframe = pd.read_csv(stn_fil)
    stn_dates = stn_dframe['Date']
    stn_dframe['DateTime'] = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in stn_dates] 
    stn_dframe = stn_dframe.set_index("DateTime")
    stn_dframe = stn_dframe.reindex(date_array, fill_value=np.nan)
    stn_stemp = stn_dframe['Spatial Avg']
    stn_stemp_master.append(stn_stemp)
    
    JRA_stemp_fi = ''.join([prod_dir,'JRA55.nc'])
    JRA_stemp_fil = xr.open_dataset(JRA_stemp_fi)
    JRA_stemp = JRA_stemp_fil['Soil_Temp'] - 273.15
    JRA_stemp = JRA_stemp.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    JRA_stemp_master.append(JRA_stemp.values.tolist())
    #print(JRA_stemp_master)
    MERRA2_stemp_fi = ''.join([prod_dir,'MERRA2.nc'])
    MERRA2_stemp_fil = xr.open_dataset(MERRA2_stemp_fi)
    MERRA2_stemp = MERRA2_stemp_fil['Soil_Temp_L1'] - 273.15
    MERRA2_stemp = MERRA2_stemp.sel(lat=lat_cen, lon=lon_cen, method='nearest')    
    MERRA2_stemp_master.append(MERRA2_stemp.values.tolist())
    #print(MERRA2_stemp)
    GLDAS_stemp_fi = ''.join([prod_dir,'GLDAS.nc'])
    GLDAS_stemp_fil = xr.open_dataset(GLDAS_stemp_fi)
    GLDAS_stemp = GLDAS_stemp_fil['Soil_Temp_L1'] - 273.15
    GLDAS_stemp = GLDAS_stemp.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    GLDAS_stemp_master.append(GLDAS_stemp.values.tolist())
      
    SF_MERRA2_rnys2 = SF_MERRA2_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    SF_MERRA2_rnys_master.append(SF_MERRA2_rnys2.values.tolist())
    SF_GLDAS_rnys2= SF_GLDAS_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    SF_GLDAS_rnys_master.append(SF_GLDAS_rnys2.values.tolist())
    SNR_JRA_rnys2 = SNR_JRA_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    SNR_JRA_rnys_master.append(SNR_JRA_rnys2.values.tolist())
    SNR_MERRA2_rnys2 = SNR_MERRA2_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    SNR_MERRA2_rnys_master.append(SNR_MERRA2_rnys2.values.tolist())
    SNR_GLDAS_rnys2 = SNR_GLDAS_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    SNR_GLDAS_rnys_master.append(SNR_GLDAS_rnys2.values.tolist())
    EV_JRA_rnys2 = EV_JRA_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    EV_JRA_rnys_master.append(EV_JRA_rnys2.values.tolist())
    EV_MERRA2_rnys2 = EV_MERRA2_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    EV_MERRA2_rnys_master.append(EV_MERRA2_rnys2.values.tolist())
    EV_GLDAS_rnys2 = EV_GLDAS_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    EV_GLDAS_rnys_master.append(EV_GLDAS_rnys2.values.tolist())
    fMSE_JRA_rnys2 = fMSE_JRA_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    JRA_fMSE_rnys_master.append(fMSE_JRA_rnys2.values.tolist())
    fMSE_MERRA2_rnys2 = fMSE_MERRA2_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    MERRA2_fMSE_rnys_master.append(fMSE_MERRA2_rnys2.values.tolist())
    fMSE_GLDAS_rnys2 = fMSE_GLDAS_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    GLDAS_fMSE_rnys_master.append(fMSE_GLDAS_rnys2.values.tolist())
    JRA_wght_rnys2 = JRA_wght_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    JRA_wght_rnys_master.append(JRA_wght_rnys2.values.tolist())
    MERRA2_wght_rnys2 = MERRA2_wght_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    MERRA2_wght_rnys_master.append(MERRA2_wght_rnys2.values.tolist())
    GLDAS_wght_rnys2 = GLDAS_wght_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    GLDAS_wght_rnys_master.append(GLDAS_wght_rnys2.values.tolist())
    TC_blended_rnys2 = TC_blended_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    TC_blend_rnys_master.append(TC_blended_rnys2.values.tolist())
    naive_blended_rnys2 = naive_blended_rnys.sel(lat=lat_cen, lon=lon_cen, method='nearest')
    naive_blend_rnys_master.append(naive_blended_rnys2.values.tolist())

#print(SF_MERRA2_rnys_master)
date_master = [j for sub in date_master for j in sub]
#gcell_master = [j fro sub in gcell_master for j in sub]
gcell_master_blended = [j for sub in gcell_master_blended for j in sub]
lat_cen_master = [j for sub in lat_cen_master for j in sub]
lon_cen_master = [j for sub in lon_cen_master for j in sub]
stn_stemp_master = [j for sub in stn_stemp_master for j in sub]
#TC_blend_csv_master = [j for sub in TC_blend_csv_master for j in sub]
#naive_blend_csv_master = [j for sub in naive_blend_csv_master for j in sub]
#TC_blend_rnys_master = [j for sub in TC_blend_rnys_master for j in sub]
#naive_blend_rnys_master = [j for sub in naive_blend_rnys_master for j in sub]
JRA_stemp_master = [j for sub in JRA_stemp_master for j in sub]
#JRA_stemp_master = [j for sub in JRA_stemp_master for j in sub]
#JRA_stemp_master = [j for sub in JRA_stemp_master for j in sub]
MERRA2_stemp_master = [j for sub in MERRA2_stemp_master for j in sub]
#MERRA2_stemp_master = [j for sub in MERRA2_stemp_master for j in sub]
#MERRA2_stemp_master = [j for sub in MERRA2_stemp_master for j in sub]
GLDAS_stemp_master = [j for sub in GLDAS_stemp_master for j in sub]
#GLDAS_stemp_master = [j for sub in GLDAS_stemp_master for j in sub]
#GLDAS_stemp_master = [j for sub in GLDAS_stemp_master for j in sub]

print(JRA_stemp_master)

TC_metrics_dframe_final = pd.DataFrame(data=gcell_master, columns=['Grid Cell'])
TC_metrics_dframe_final['Central Lat'] = lat_master
TC_metrics_dframe_final['Central Lon'] = lon_master
print(TC_metrics_dframe_final)

TC_metrics_dframe_final['SF MERRA2 cdo'] = SF_MERRA2_cdo_master
TC_metrics_dframe_final['SF MERRA2 ncks'] = SF_MERRA2_ncks_master
TC_metrics_dframe_final['SF MERRA2 rnys'] = SF_MERRA2_rnys_master
TC_metrics_dframe_final['SF GLDAS cdo'] = SF_GLDAS_cdo_master
TC_metrics_dframe_final['SF GLDAS ncks'] = SF_GLDAS_ncks_master
TC_metrics_dframe_final['SF GLDAS rnys'] = SF_GLDAS_rnys_master

TC_metrics_dframe_final['Err Var JRA55 cdo'] = EV_JRA_cdo_master
TC_metrics_dframe_final['Err Var JRA55 ncks'] = EV_JRA_ncks_master
TC_metrics_dframe_final['Err Var JRA55 rnys'] = EV_JRA_rnys_master
TC_metrics_dframe_final['Err Var MERRA2 cdo'] = EV_MERRA2_cdo_master
TC_metrics_dframe_final['Err Var MERRA2 ncks'] = EV_MERRA2_ncks_master
TC_metrics_dframe_final['Err Var MERRA2 rnys'] = EV_MERRA2_rnys_master
TC_metrics_dframe_final['Err Var GLDAS cdo'] = EV_GLDAS_cdo_master
TC_metrics_dframe_final['Err Var GLDAS ncks'] = EV_GLDAS_ncks_master
TC_metrics_dframe_final['Err Var GLDAS rnys'] = EV_GLDAS_rnys_master

TC_metrics_dframe_final['SNR JRA55 cdo'] = SNR_JRA_cdo_master
TC_metrics_dframe_final['SNR JRA55 ncks'] = SNR_JRA_ncks_master
TC_metrics_dframe_final['SNR JRA55 rnys'] = SNR_JRA_rnys_master
TC_metrics_dframe_final['SNR MERRA2 cdo'] = SNR_MERRA2_cdo_master
TC_metrics_dframe_final['SNR MERRA2 ncks'] = SNR_MERRA2_ncks_master
TC_metrics_dframe_final['SNR MERRA2 rnys'] = SNR_MERRA2_rnys_master
TC_metrics_dframe_final['SNR GLDAS cdo'] = SNR_GLDAS_cdo_master
TC_metrics_dframe_final['SNR GLDAS ncks'] = SNR_GLDAS_ncks_master
TC_metrics_dframe_final['SNR GLDAS rnys'] = SNR_GLDAS_rnys_master

TC_metrics_dframe_final['wght JRA55 cdo'] = JRA_wght_cdo_master
TC_metrics_dframe_final['wght JRA55 ncks'] = JRA_wght_ncks_master
TC_metrics_dframe_final['wght JRA55 rnys'] = JRA_wght_rnys_master
TC_metrics_dframe_final['wght MERRA2 cdo'] = MERRA2_wght_cdo_master
TC_metrics_dframe_final['wght MERRA2 ncks'] = MERRA2_wght_ncks_master
TC_metrics_dframe_final['wght MERRA2 rnys'] = MERRA2_wght_rnys_master
TC_metrics_dframe_final['wght GLDAS cdo'] = GLDAS_wght_cdo_master
TC_metrics_dframe_final['wght GLDAS ncks'] = GLDAS_wght_ncks_master
TC_metrics_dframe_final['wght GLDAS rnys'] = GLDAS_wght_rnys_master

#print(TC_metrics_dframe_final)

ofil = "/mnt/data/users/herringtont/soil_temp/global_triple_collocation/compare_metrics/raw_temp/remapnn_compare_metrics_ncks_cdo.csv"
TC_metrics_dframe_final.to_csv(ofil, index=False)

#blended_dframe_final = pd.DataFrame(data=date_master, columns=['Date'])
#blended_dframe_final['Grid Cell'] = gcell_master_blended
#blended_dframe_final['Central Lat'] = lat_cen_master
#blended_dframe_final['Central Lon'] = lon_cen_master
#blended_dframe_final['Station'] = stn_stemp_master
#blended_dframe_final['TC Blend RNYS'] = TC_blend_rnys_master
#blended_dframe_final['TC Blend CSV'] = TC_blend_csv_master
#blended_dframe_final['Naive Blend RNYS'] = naive_blend_rnys_master
#blended_dframe_final['Naive Blend CSV'] = naive_blend_csv_master
#blended_dframe_final['JRA55'] = JRA_stemp_master
#blended_dframe_final['MERRA2'] = MERRA2_stemp_master
#blended_dframe_final['GLDAS'] = GLDAS_stemp_master
#
#print(blended_dframe_final)
#
#gcell_uq = np.unique(gcell_master_blended)
#num_gcell = len(gcell_uq)
#
#
#
###################### create 2 figures (with a maximum of 12 subplots each) #######################
#ymin = -40
#ymax = 40
#xmin = np.datetime64(datetime.date(1980,1,1),'Y')
#xmax = np.datetime64(datetime.date(2019,1,1),'Y')
#
#for j in range (0,2): ##create 2 figures
#    fig = plt.figure()
#    fig,axs = plt.subplots(nrows = 4, ncols = 3, sharex = 'col', sharey = 'row', figsize=(20,18))
#
#    if (j == 0): ##first figure has 12 subplots
#    	st = 0
#    	ed = 12
#
#    elif(j == 1):
#    	st = 12
#    	ed = 23
#	
#    n = 1 #figure number	
##### loop through 23 grid cells and grab data to plot ####
#    for k in range(st,ed):    
#    	gcell_k = gcell_uq[k]
#    	gcell_st = gcell_uq[st]
#    	gcell_ed = gcell_uq[ed-1]
#    	dframe_gcell = blended_dframe_final[blended_dframe_final['Grid Cell'] == gcell_k]
#    	date_gcell = dframe_gcell['Date']
#    	date_gcell2 = [datetime.datetime.strptime(x,'%Y-%m-%d') for x in date_gcell]	
#    	lat = dframe_gcell['Central Lat'].iloc[0]
#    	lat2 = round(lat,2)
#    	lon = dframe_gcell['Central Lon'].iloc[0]
#    	lon2 = round(lon,2)
#    	stn_stemp2 = dframe_gcell['Station'].values		
#    	TC_blend_stemp_rnys = dframe_gcell['TC Blend RNYS'].values
#    	TC_blend_stemp_csv = dframe_gcell['TC Blend CSV'].values
#    	naive_blend_stemp_rnys = dframe_gcell['Naive Blend RNYS'].values
#    	naive_blend_stemp_csv = dframe_gcell['Naive Blend CSV'].values
#    	JRA_stemp = dframe_gcell['JRA55'].values
#    	MERRA2_stemp = dframe_gcell['MERRA2'].values
#    	GLDAS_stemp = dframe_gcell['GLDAS'].values
#
#    	ax = plt.subplot(4,3,n)
#    	ax.plot(date_gcell2,stn_stemp2,label='Station',marker='o',markerfacecolor='dodgerblue',markersize=2,color='royalblue')
#    	ax.plot(date_gcell2,TC_blend_stemp_rnys,label='TC Blend (netCDF)',marker='s',markerfacecolor='chartreuse',markersize=2,color='lawngreen')	
#    	ax.plot(date_gcell2,TC_blend_stemp_csv,label='TC Blend (CSV)',marker='s',markerfacecolor='olivedrab',markersize=2,color='forestgreen')
#    	ax.plot(date_gcell2,naive_blend_stemp_rnys,label='Naive Blend (netCDF)',marker='v',markerfacecolor='coral',markersize=2,color='orangered')
#    	ax.plot(date_gcell2,naive_blend_stemp_csv,label='Naive Blend (CSV)',marker='v',markerfacecolor='firebrick',markersize=2,color='darkred')
#    	ax.plot(date_gcell2,JRA_stemp,label='JRA55',marker='*',markerfacecolor='plum',markersize=2,color='mediumorchid')	
#    	ax.plot(date_gcell2,MERRA2_stemp,label='MERRA2',marker='D',markerfacecolor='gold',markersize=2,color='goldenrod')	
#    	ax.plot(date_gcell2,MERRA2_stemp,label='GLDAS',marker='x',markerfacecolor='lightslategrey',markersize=2,color='slategrey')
#
#    	#ax.xaxis.set_major_locator(MultipleLocator(5)) #major tick every 5 years
#    	#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')) #only show the year
#    	#ax.xaxis.set_minor_locator(MultipleLocator(1)) #minor tick every year   					
#    	ax.yaxis.set_major_locator(MultipleLocator(5)) #every 5 degrees will be a major tick
#    	ax.yaxis.set_minor_locator(MultipleLocator(1)) #every 1 degrees will be a minor tick
#
#    	fig.autofmt_xdate()
#    	ax.fmt_xdata = mdates.DateFormatter('%Y')
#    	ax.set_ylim(ymin,ymax)
#    	ax.text(0.05,0.95,s='Grid Cell: '+str(gcell_k)+', Lat: '+str(lat2)+'$^\circ$N, Lon :'+str(lon2)+'$^\circ$',transform=ax.transAxes)
#    	#ax.legend() 
#    	lines = []
#    	labels = []
#    	for ax in fig.get_axes():
#    		axLine, axLabel = ax.get_legend_handles_labels()
#    		lines.extend(axLine)
#    		labels.extend(axLabel)
#
#    	n = n+1
#
#    if (i == 1):
#    	for k in range(st,ed):
#    		plt.subplot(4,3,k).set_visible(False)
#
#   			
#    fig.add_subplot(111, frameon=False) #create large subplot which will include the plot labels for plots
#    plt.tick_params(labelcolor='none',bottom=False,left=False) #hide ticks
#    plt.xlabel('Date',fontweight='bold')
#    plt.ylabel('Soil Temp ($^\circ$ C)',fontweight='bold')
#    fig.legend(lines[0:7],labels[0:7],loc="right",title="Legend")
#
#    if ( i == 0): 
#    	plt.tight_layout()
#
#    plt_fil = ''.join(['/mnt/data/users/herringtont/soil_temp/plots/temp_timeseries/raw/grid_'+str(gcell_st)+'_'+str(gcell_ed)+'.png'])
#    print(plt_fil)
#    plt.savefig(plt_fil)
#    plt.close()
#
