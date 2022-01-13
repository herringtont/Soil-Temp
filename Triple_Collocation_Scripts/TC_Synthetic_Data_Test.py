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
from calendar import isleap
from dateutil.relativedelta import *
from pathlib import Path
import pytesmo.scaling as scaling
import pytesmo.metrics as metrics

########################### Create Synthetic Test Data #############################

# number of observations
n = 1000000
# x coordinates for initializing the sine curve
coord = np.linspace(0, 2*np.pi, n)
signal = np.sin(coord)

# error i.e. epsilon of the three synthetic time series
sig_err_x = 0.02
sig_err_y = 0.07
sig_err_z = 0.04
err_x = np.random.normal(0, sig_err_x, n)
err_y = np.random.normal(0, sig_err_y, n)
err_z = np.random.normal(0, sig_err_z, n)

# additive and multiplicative biases
# they are assumed to be zero for dataset x
alpha_y_syn = 0.2
alpha_z_syn = 0.5

beta_y_syn = 0.9
beta_z_syn = 1.6

x = signal + err_x
# here we assume errors that are already scaled
y = alpha_y_syn + beta_y_syn * (signal + err_y)
z = alpha_z_syn + beta_z_syn * (signal + err_z)

e_x_true = 0.0200
e_y_true = 0.0700
e_z_true = 0.0400

y_scaled = scaling.mean_std(y,x)
z_scaled = scaling.mean_std(z,x)

e_x, e_y, e_z = metrics.tcol_error(x, y_scaled, z_scaled)
print("***Pytesmo Formulation***")
print("***Approach 1 - Scaling***")
print(e_x**2, e_y**2, e_z**2)

snr,err,beta = metrics.tcol_snr(x,y,z)
print("***Approach 2 - Covariances***")
print(err[0]**2,err[1]**2,err[2]**2)
print("Scaling Parameters Estimated")
print(1/beta[1],1/beta[2])
print("SNR (dB) estimated")
print(snr[0],snr[1],snr[2])
########################### APPROACH 1 (SCALING) ################################

x_df = x - np.mean(x)  #### mean value of timeseries   				
y_df = y - np.mean(y)  #### mean value of timeseries
z_df = z - np.mean(z) #### mean value of timeseries		
				
beta_ystar = np.mean(x_df*z_df)/np.mean(y_df*z_df) ##rescaling factor for Y
beta_zstar = np.mean(x_df*y_df)/np.mean(z_df*y_df) ##rescaling factor for Z

y_factor = 1/beta_ystar
z_factor = 1/beta_zstar

x_bar = np.mean(x)
y_bar = np.mean(y)
z_bar = np.mean(z)

y_diff = y-y_bar
z_diff = z-z_bar

y_rescaled = (beta_ystar*y_diff)+x_bar
z_rescaled = (beta_zstar*z_diff)+x_bar   				

err_varx_scaled = np.mean((x-y_rescaled)*(x-z_rescaled)) ## error variance of x using difference notation
err_vary_scaled = np.mean((y_rescaled-x)*(y_rescaled-z_rescaled)) ## error variance of y using difference notation
err_varz_scaled = np.mean((z_rescaled-x)*(z_rescaled-y_rescaled)) ## error variance of z using difference notation

err_stdx_scaled = math.sqrt(err_varx_scaled)
err_stdy_scaled = math.sqrt(err_vary_scaled)
err_stdz_scaled = math.sqrt(err_varz_scaled)

print("***Tyler's Script***")				   				
print("***Approach 1 - Scaling***")
print("Error Variances:")
print(err_varx_scaled,err_vary_scaled,err_varz_scaled)
print("Scaling Factors:")
print(y_factor,z_factor)


########################## APPROACH 2 (COVARIANCES) #############################
x_std = np.std(x)
y_std = np.std(y)
z_std = np.std(z)

signal_varx = (np.cov(x,y)[0][1]*np.cov(x,z)[0][1])/np.cov(y,z)[0][1] ###Signal to Noise Ratio of X (soil temperature sensitivity of the data set) 
signal_vary = (np.cov(y,x)[0][1]*np.cov(y,z)[0][1])/np.cov(x,z)[0][1] ###Signal to Noise Ratio of Y (soil temperature sensitivity of the data set) 
signal_varz = (np.cov(z,x)[0][1]*np.cov(z,y)[0][1])/np.cov(x,y)[0][1] ###Signal to Noise Ratio of Z (soil temperature sensitivity of the data set)

err_varx = np.var(x) - signal_varx ##Error variance of dataset X using covariance notation
err_vary = np.var(y) - signal_vary ##Error variance of dataset Y using covariance notation
err_varz = np.var(z) - signal_varz ##Error variance of dataset Z using covariance notation

snrx = signal_varx/err_varx    				
snry = signal_vary/err_vary
snrz = signal_varz/err_varz 

snrx_log = 10*math.log10(snrx)
snry_log = 10*math.log10(snry)
snrz_log = 10*math.log10(snrz)
				
nsrx = err_varx/signal_varx ##Noise to Signal Ratio of dataset x
nsry = err_vary/signal_vary ##Noise to Signal Ratio of dataset y
nsrz = err_varz/signal_varz ##Noise to Signal Ratio of dataset z

Rxy = 1/math.sqrt((1+nsrx)*(1+nsry)) ##Pearson correlation between dataset X and dataset Y
Ryz = 1/math.sqrt((1+nsry)*(1+nsrz)) ##Pearson correlation between dataset Y and dataset Z
Rxz = 1/math.sqrt((1+nsrx)*(1+nsrz)) ##Pearson correlation between dataset X and dataset Z

beta_y = np.cov(y,z)[0][1]/np.cov(x,z)[0][1]
beta_z = np.cov(y,z)[0][1]/np.cov(x,y)[0][1]
scaling_factor_Y = beta_y
scaling_factor_Z = beta_z

print("***Approach 2 - Covariance***")
print("Error Variances:")
print(err_varx,err_vary,err_varz)
print("Scaling Factor of Y, Scaling Factor of Z:")
print(scaling_factor_Y, scaling_factor_Z)
print("SNR (dB)")
print(snrx_log,snry_log,snrz_log)

y_beta_scaled = y * beta_y
z_beta_scaled = z * beta_z

y_ab_scaled = y_beta_scaled - np.mean(y_beta_scaled)
z_ab_scaled = z_beta_scaled - np.mean(z_beta_scaled)

print("Rxy, Ryz, and Rxz:")
print(Rxy,Ryz,Rxz)

print("Rx, Ry and Rz:")

Rx = math.sqrt(snrx/(1+snrx)) ##Correlation between Dataset X and true soil temp 
Ry = math.sqrt(snry/(1+snry)) ##Correlation between Dataset X and true soil temp 
Rz = math.sqrt(snrz/(1+snrz))
			
print(Rx, Ry, Rz)

print("fRMSE:")
fRMSE_x = math.sqrt(1/(1+snrx))
fRMSE_y = math.sqrt(1/(1+snry))
fRMSE_z = math.sqrt(1/(1+snrz))
print(fRMSE_x, fRMSE_y, fRMSE_z)
