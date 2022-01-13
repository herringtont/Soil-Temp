import cdo
from cdo import Cdo
import os
import subprocess
from subprocess import call
import cdo
import csv
import datetime as dt  # Python standard library datetime  module
import numpy as np
import pandas as pd
import scipy
import netCDF4
from netCDF4 import Dataset, num2date  # http://code.google.com/p/netcdf4-python/
import matplotlib.pyplot as plt
import math
import pytesmo
import xarray as xr

#set file locations
JRAfi = str("/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/site_level/JRA55/JRA55_site_1.nc")

#Open netCDF files with xarray
JRAfil = xr.open_dataset(JRAfi)

#extract soil temperatures
JRA_stemp = JRAfil.Soil_Temp.isel(lat=0,lon=0)
#JRA_stemp2 = JRA_stemp.round(decimals=2)
JRA_stemp.attrs = JRAfil.Soil_Temp.attrs
#JRA_stemp.attrs["units"] = "deg C"
JRA_s = np.array(JRA_stemp)



#create artificial data for columns 2 and 3
JRA_s3 = np.sqrt(JRA_s)
JRA_s2 = JRA_s**2


x = JRA_s
y = JRA_s2
z = JRA_s3

################################################
###### APPROACH 1 (SCALING)

def mean_std(src, ref):
    return ((src - np.nanmean(src)) /
            np.nanstd(src)) * np.nanstd(ref) + np.nanmean(ref)

def tcol_error(x, y, z):
    e_x = np.sqrt(np.abs(np.nanmean((x - y) * (x - z))))
    e_y = np.sqrt(np.abs(np.nanmean((y - x) * (y - z))))
    e_z = np.sqrt(np.abs(np.nanmean((z - x) * (z - y))))

    return e_x, e_y, e_z

x = JRA_s
y_scaled = mean_std(JRA_s2, JRA_s)
z_scaled = mean_std(JRA_s3, JRA_s)

e_x, e_y, e_z = tcol_error(x, y_scaled, z_scaled)
print("***Approach 1 - Scaling***")
print("Errors:")
print(e_x,e_y,e_z)


################################################
###### APPROACH 2 (COVARIANCES)

def triple_collocation_snr(x, y, z, ref_ind=0):
    cov = np.cov(np.vstack((x, y, z)))
    ind = (0, 1, 2, 0, 1, 2)
    no_ref_ind = np.where(np.arange(3) != ref_ind)[0]
    
    snr = -10 * np.log10([abs(((cov[i, i] * cov[ind[i + 1], ind[i + 2]]) / (cov[i, ind[i + 1]] * cov[i, ind[i + 2]])) - 1)
                         for i in np.arange(3)])
    err_var = np.array([
        abs(cov[i, i] -
        (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]])
        for i in np.arange(3)])

    beta = np.array([cov[ref_ind, no_ref_ind[no_ref_ind != i][0]] /
                     cov[i, no_ref_ind[no_ref_ind != i][0]] if i != ref_ind
                     else 1 for i in np.arange(3)])

    return snr, np.sqrt(err_var) * beta, beta

snr, err, beta = triple_collocation_snr(JRA_s, JRA_s2, JRA_s3)

print("***Approach 2 - Covariance***")
print("Signal to Noise Ratios:")
print(snr)
print("Errors:")
print(err)
print("Inverse of Beta Y, Beta Z:")
print(1/beta[1], 1/beta[2])

y_beta_scaled = y * beta[1]
z_beta_scaled = z * beta[2]

y_ab_scaled = y_beta_scaled - np.mean(y_beta_scaled)
z_ab_scaled = z_beta_scaled - np.mean(z_beta_scaled)

print("R")
R_xy = 1.0 / math.sqrt((1.0+(1.0/snr[0]))*(1.0+(1.0/snr[1])))
R_yz = 1.0 / math.sqrt((1.0+(1.0/snr[1]))*(1.0+(1.0/snr[2])))
R_xz = 1.0 / math.sqrt((1.0+(1.0/snr[0]))*(1.0+(1.0/snr[2])))
print(R_xy, R_yz, R_xz)

print("fRMSE")
fRMSE_x = 1.0 / (1.0 + snr[0])
fRMSE_y = 1.0 / (1.0 + snr[1])
fRMSE_z = 1.0 / (1.0 + snr[2])
print(fRMSE_x, fRMSE_y, fRMSE_z)
