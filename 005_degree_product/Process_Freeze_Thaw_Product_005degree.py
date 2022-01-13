import os
import glob
import netCDF4
import csv
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import pathlib
import dask
from multiprocessing.pool import ThreadPool
from dask.diagnostics import ProgressBar
dask.config.set(pool=ThreadPool(4))
dask.config.set(**{'array.slicing.split_large_chunks': True})

############ Set Directories and Variables ##############

rmp_type = ['remapbil','remapnn']
#rmp_type = ['remapbil']
layers = ['0_49.9','50_99.9','100_200.0']
#layers= ['100_200.0']
stemp_var = 'TSLB'
common_date_index = pd.date_range(start='1970-01-01',end='2021-09-01', freq='MS')
new_date_index = pd.date_range(start='2000-01-01',end='2016-12-01', freq='MS')


#chunks_ASR = {'Time':1,'lat':1200,'lon':7200}
#chunks_other = {'time':1,lat':1200,'lon':7200}

chunks_ASR = {'Time':1,'lat':1200,'lon':7200}
chunks_other = {'time':1,'lat':1200,'lon':7200}
######### Grab Data ########

for rmp_i in rmp_type: #loop through remap styles

    wkdir = ''.join(['/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/005_degree/'+str(rmp_i)+'/soil_temp/sellonlatbox/layers/'])
    
    for layer_j in layers:

   	
    	if (layer_j == '0_49.9'):
    		sublayer_ASR_file1 = ''.join([wkdir+'ASR_stl1.nc'])
    		sublayer_ASR_file2 = ''.join([wkdir+'ASR_stl2.nc'])
    		sublayer_ERA5_Land_file1 = ''.join([wkdir+'ERA5_Land_stl1.nc'])
    		sublayer_ERA5_Land_file2 = ''.join([wkdir+'ERA5_Land_stl2.nc'])    	
    		sublayer_FLDAS_file1 = ''.join([wkdir+'FLDAS_stl1.nc'])
    		sublayer_FLDAS_file2 = ''.join([wkdir+'FLDAS_stl2.nc'])

    		sublayer1_ASR = xr.open_dataset(sublayer_ASR_file1,chunks=chunks_ASR)
    		sublayer2_ASR = xr.open_dataset(sublayer_ASR_file2,chunks=chunks_ASR)

    		sublayer1_ERA5_Land = xr.open_dataset(sublayer_ERA5_Land_file1,chunks=chunks_other)
    		sublayer2_ERA5_Land = xr.open_dataset(sublayer_ERA5_Land_file2,chunks=chunks_other)

    		sublayer1_FLDAS = xr.open_dataset(sublayer_FLDAS_file1,chunks=chunks_other)
    		sublayer2_FLDAS = xr.open_dataset(sublayer_FLDAS_file2,chunks=chunks_other)


    		sublayer1_ASR = sublayer1_ASR.rename({'Time':'time'})
    		sublayer1_ASR['time'] = new_date_index #reindex dates to first of month
    		sublayer1_ASR = sublayer1_ASR.rename({'TSLB':'soil_temp_ASR'}) #rename TSLB variable to 'soil_temp_ASR'

    		sublayer2_ASR = sublayer2_ASR.rename({'Time':'time'})
    		sublayer2_ASR['time'] = new_date_index #reindex dates to first of month
    		sublayer2_ASR = sublayer2_ASR.rename({'TSLB':'soil_temp_ASR'}) #rename TSLB variable to 'soil_temp_ASR'


    		sublayer1_ERA5_Land = sublayer1_ERA5_Land.rename({'TSLB':'soil_temp_ERA5'})
    		sublayer2_ERA5_Land = sublayer2_ERA5_Land.rename({'TSLB':'soil_temp_ERA5'})

    		sublayer1_FLDAS = sublayer1_FLDAS.rename({'TSLB':'soil_temp_FLDAS'})
    		sublayer2_FLDAS = sublayer2_FLDAS.rename({'TSLB':'soil_temp_FLDAS'})

    	elif (layer_j == '50_99.9'):
    		layer_ASR_file = ''.join([wkdir+'ASR_stl3.nc'])
    		layer_ERA5_Land_file = ''.join([wkdir+'ERA5_Land_stl3.nc'])
    		layer_FLDAS_file = ''.join([wkdir+'FLDAS_stl3.nc'])    
 
     
    	elif (layer_j == '100_200.0'):
    		layer_ASR_file = ''.join([wkdir+'ASR_stl4.nc'])
    		layer_ERA5_Land_file = ''.join([wkdir+'ERA5_Land_stl4.nc'])
    		layer_FLDAS_file = ''.join([wkdir+'FLDAS_stl4.nc'])     



    	if (layer_j == '50_99.9' or layer_j == '100_200.0'):

    		layer_ASR = xr.open_dataset(layer_ASR_file,chunks=chunks_ASR)
    		layer_ERA5_Land = xr.open_dataset(layer_ERA5_Land_file,chunks=chunks_other)
    		layer_FLDAS = xr.open_dataset(layer_FLDAS_file,chunks=chunks_other)

    		#layer_ASR =layer_ASR.rename({'Time':'time'})
    		layer_ASR = layer_ASR.rename({'Time':'time'})
    		layer_ASR['time'] = new_date_index #reindex dates to first of month
    		layer_ASR = layer_ASR.rename({'TSLB':'soil_temp_ASR'}) #rename TSLB variable to 'soil_temp_ASR'

    		layer_ERA5_Land = layer_ERA5_Land.rename({'TSLB':'soil_temp_ERA5'})
    		layer_FLDAS = layer_FLDAS.rename({'TSLB':'soil_temp_FLDAS'})  		


    #### merge files together to calculate ensemble mean ####
    

    		stemp_ASR = layer_ASR['soil_temp_ASR']
    		stemp_ERA5_Land = layer_ERA5_Land['soil_temp_ERA5']
    		stemp_FLDAS = layer_FLDAS['soil_temp_FLDAS']
    		stemp_ASR_reindex = stemp_ASR.reindex_like(stemp_ERA5_Land, fill_value=np.nan)
    		stemp_FLDAS_reindex = stemp_FLDAS.reindex_like(stemp_ERA5_Land, fill_value=np.nan)

    		stemp_all = xr.concat([stemp_ASR_reindex,stemp_ERA5_Land,stemp_FLDAS_reindex],pd.Index(['ASR','ERA5-Land','FLDAS'],name='ensemble'),fill_value=np.nan) #merge datasets together

    		stemp_ensmean = stemp_all.mean('ensemble',skipna=True)
    		stemp_ensmean = stemp_ensmean.to_dataset(name='soil_temp_EnsMean')
    		chunks_out = {'time':1,'lat':300,'lon':360}
    		stemp_ensmean = stemp_ensmean.chunk(chunks=chunks_out)

    		stemp_ensmean.attrs = stemp_ERA5_Land.attrs 
    		print(stemp_ensmean)

    		#chunks_nc = (1,1200,7200)
    		#var_enc = dict(zlib=True, complevel=1, _FillValue=-9999, chunksizes=chunks_nc)
    		var_enc = dict(zlib=True, complevel=1, _FillValue=-9999)

    		ensmean_netcdf_fil = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/005_degree/'+str(rmp_i)+'/soil_temp/sellonlatbox/layers/ensmean_'+str(rmp_i)+'_'+str(layer_j)+'.nc'

    		with ProgressBar():
    			stemp_ensmean.to_netcdf(ensmean_netcdf_fil,unlimited_dims=['time'],engine='netcdf4',encoding={'soil_temp_EnsMean':var_enc})    		


    	elif (layer_j == '0_49.9'):

    		stemp_ASR_sublayer1 = sublayer1_ASR['soil_temp_ASR']
    		stemp_ASR_sublayer2 = sublayer2_ASR['soil_temp_ASR']

    		stemp_ERA5_Land_sublayer1 = sublayer1_ERA5_Land['soil_temp_ERA5']
    		stemp_ERA5_Land_sublayer2 = sublayer2_ERA5_Land['soil_temp_ERA5']

    		stemp_FLDAS_sublayer1 = sublayer1_FLDAS['soil_temp_FLDAS']
    		stemp_FLDAS_sublayer2 = sublayer2_FLDAS['soil_temp_FLDAS']


    #### Determine which sublayer is colder ####

    		    		
    		ASR_all_sub = xr.concat([stemp_ASR_sublayer1,stemp_ASR_sublayer2],pd.Index(['0cm-25cm','25cm-50cm'],name='sublayer'),fill_value=np.nan) #concatenate sublayers together with new dummy dimension to calculate minimum along
    		ASR_min = ASR_all_sub.min('sublayer',skipna=True) #calculate minimum temperature of the 2 sublayers

    		ERA5_all_sub = xr.concat([stemp_ERA5_Land_sublayer1,stemp_ERA5_Land_sublayer2],pd.Index(['0cm-25cm','25cm-50cm'],name='sublayer'),fill_value=np.nan) #concatenate sublayers together with new dummy dimension to calculate minimum along
    		ERA5_min = ERA5_all_sub.min('sublayer',skipna=True) #calculate minimum temperature of the 2 sublayers

    		FLDAS_all_sub = xr.concat([stemp_FLDAS_sublayer1,stemp_FLDAS_sublayer2],pd.Index(['0cm-25cm','25cm-50cm'],name='sublayer'),fill_value=np.nan) #concatenate sublayers together with new dummy dimension to calculate minimum along
    		FLDAS_min = FLDAS_all_sub.min('sublayer',skipna=True) #calculate minimum temperature of the 2 sublayers


    		stemp_ASR_reindex = ASR_min.reindex_like(stemp_ERA5_Land_sublayer1, fill_value=np.nan)
    		stemp_FLDAS_reindex = FLDAS_min.reindex_like(stemp_ERA5_Land_sublayer1, fill_value=np.nan)
    		stemp_ERA5_Land = ERA5_min.reindex_like(stemp_ERA5_Land_sublayer1, fill_value=np.nan)

    #### merge files together to calculate ensemble mean ####
 
    		stemp_all = xr.concat([stemp_ASR_reindex,stemp_ERA5_Land,stemp_FLDAS_reindex],pd.Index(['ASR','ERA5-Land','FLDAS'],name='ensemble'),fill_value=np.nan) #merge datasets together

    		stemp_ensmean = stemp_all.mean('ensemble',skipna=True)
    		stemp_ensmean = stemp_ensmean.to_dataset(name='soil_temp_EnsMean')
    		chunks_out = {'time':1,'lat':300,'lon':360}
    		stemp_ensmean = stemp_ensmean.chunk(chunks=chunks_out)

    		stemp_ensmean.attrs = stemp_ERA5_Land.attrs 
    		print(stemp_ensmean)

    		#chunks_nc = (1,1200,7200)
    		#var_enc = dict(zlib=True, complevel=1, _FillValue=-9999, chunksizes=chunks_nc)
    		var_enc = dict(zlib=True, complevel=1, _FillValue=-9999)

    		ensmean_netcdf_fil = '/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/005_degree/'+str(rmp_i)+'/soil_temp/sellonlatbox/layers/ensmean_'+str(rmp_i)+'_'+str(layer_j)+'.nc'

    		with ProgressBar():
    			stemp_ensmean.to_netcdf(ensmean_netcdf_fil,unlimited_dims=['time'],engine='netcdf4',encoding={'soil_temp_EnsMean':var_enc}) 
