import os
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import xarray as xr
import matplotlib.pyplot as plt
import scipy
import cdo
from cdo import Cdo
import re
import math
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import math
import cftime
from calendar import isleap
from dateutil.relativedelta import *
from scipy import stats

######## Set Directories ########
CFSR_dir = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/CFSR_stats/anom_2/site_level/"
CFSR2_dir = "/mnt/data/users/herringtont/soil_temp/reanalysis/monthly/remap/rename/CFSR_stats/anom_2/site_level/"


######### Create Master Arrays ######
site_LY1 = []
site_LY2 = []
site_LY3 = []
site_LY4 = []

N1_LY1 = []
N1_LY2 = []
N1_LY3 = []
N1_LY4 = []

N2_LY1 = []
N2_LY2 = []
N2_LY3 = []
N2_LY4 = []

CFSR_mean_LY1 = []
CFSR_mean_LY2 = []
CFSR_mean_LY3 = []
CFSR_mean_LY4 = []
CFSR_var_LY1 = []
CFSR_var_LY2 = []
CFSR_var_LY3 = []
CFSR_var_LY4 = []

CFSR2_mean_LY1 = []
CFSR2_mean_LY2 = []
CFSR2_mean_LY3 = []
CFSR2_mean_LY4 = []
CFSR2_var_LY1 = []
CFSR2_var_LY2 = []
CFSR2_var_LY3 = []
CFSR2_var_LY4 = []

T_LY1 = []
T_LY2 = []
T_LY3 = []
T_LY4 = []

P1_LY1 = []
P1_LY2 = []
P1_LY3 = []
P1_LY4 = []

P2_LY1 = []
P2_LY2 = []
P2_LY3 = []
P2_LY4 = []

F_LY1 = []
F_LY2 = []
F_LY3 = []
F_LY4 = []

######## delineate soil layers ########
layers = ["Soil_Temp_L1","Soil_Temp_L2","Soil_Temp_L3","Soil_Temp_L4"]
for j in range(0,4):
    lyr_val = str(j+1)
    #print(lyr_val)	
    lyr = layers[j]

######### loop through files #########
    for i in range (1,300):
    	i_string = str(i)
    	CFSR_fil = "".join([CFSR_dir,"CFSR_deseasonalized_site_",i_string,".nc"])
    	CFSR2_fil = "".join([CFSR2_dir,"CFSR2_deseasonalized_site_",i_string,".nc"])
    	print("site : ",i," soil layer: ",lyr_val)

######## Read in data from netCDF file using XArray ############

    	CFSR_dat = xr.open_dataset(CFSR_fil)
    	CFSR_stemp = CFSR_dat[lyr]
    	CFSR_stemp2 = CFSR_stemp.isel(lon=0,lat=0,drop=True)  
    	CFSR2_dat = xr.open_dataset(CFSR2_fil)
    	CFSR2_stemp = CFSR2_dat[lyr]
    	CFSR2_stemp2 = CFSR2_stemp.isel(lon=0,lat=0,drop=True) 
    	#print(CFSR_stemp2)	
    	CFSRmean = CFSR_stemp.mean().values
    	CFSR_check = np.isnan(CFSR_stemp2).any()
    	CFSRvar = CFSR_stemp.var(ddof=1).values
    	CFSRs = np.sqrt(CFSRvar)
    	CFSR2mean = CFSR2_stemp.mean().values
    	CFSR2_check = np.isnan(CFSR2_stemp2).any()	
    	CFSR2var = CFSR2_stemp.var(ddof=1).values
    	CFSR2s = np.sqrt(CFSR2var)	
    	print("CFSR mean: ",CFSRmean," CFSR2 mean: ",CFSR2mean)
    	print("CFSR variance: ",CFSRvar," CFSR2 variance: ",CFSR2var)
	 		
    	if(j == 0):
    		N1 = len(CFSR_stemp2.values)
    		N2 = len(CFSR_stemp2.values)

    		if (CFSR_check == False and CFSR2_check == False): #if both CFSR and CFSR2 have data
					
			##### Conduct t-test of means ######  
    			s = np.sqrt((CFSRvar/N1)+(CFSR2var/N2))
    			t = (CFSRmean - CFSR2mean)/s   #Welch's t-statistic
    			df = ((CFSRvar/N1)+(CFSR2var/N2))**2/((CFSRvar/(N1-1))**2+(CFSR2var/(N2-1))**2)   #d.f. for unequal sample size
    			t_score, p_value = stats.ttest_ind_from_stats(mean1=CFSRmean, std1=CFSRs, nobs1=N1, mean2=CFSR2mean, std2=CFSR2s, nobs2=N2, equal_var=False)
    			#print("T-statistic: ",t,t_score) 
    			T_LY1.append(t_score)
    			P1_LY1.append(p_value)
    			N1_LY1.append(N1)
    			N2_LY1.append(N2)
									
			##### Conduct F-test of variances #####	    			
    			dfX = len(CFSR_stemp2.values) - 1
    			dfY = len(CFSR2_stemp2.values) - 1
    			F = CFSRvar/CFSR2var
    			#print(dfX,dfY,F)			
    			p_val2 = 1 - scipy.stats.f.cdf(F,dfX,dfY)
    			F_LY1.append(F)
    			P2_LY1.append(p_val2)			
    			site_LY1.append(i_string)
    			CFSR_mean_LY1.append(CFSRmean)
    			CFSR_var_LY1.append(CFSRvar)
    			CFSR2_mean_LY1.append(CFSR2mean)
    			CFSR2_var_LY1.append(CFSR2var)

    	elif(j == 1):
    		N1 = len(CFSR_stemp2.values)
    		N2 = len(CFSR_stemp2.values)

    		if (CFSR_check == False and CFSR2_check == False): #if both CFSR and CFSR2 have data
					
			##### Conduct t-test of means ######  
    			s = np.sqrt((CFSRvar/N1)+(CFSR2var/N2))
    			t = (CFSRmean - CFSR2mean)/s   #Welch's t-statistic
    			df = ((CFSRvar/N1)+(CFSR2var/N2))**2/((CFSRvar/(N1-1))**2+(CFSR2var/(N2-1))**2)   #d.f. for unequal sample size
    			t_score, p_value = stats.ttest_ind_from_stats(mean1=CFSRmean, std1=CFSRs, nobs1=N1, mean2=CFSR2mean, std2=CFSR2s, nobs2=N2, equal_var=False)
    			#print(t_score) 
    			T_LY2.append(t_score)
    			P1_LY2.append(p_value)
    			N1_LY2.append(N1)
    			N2_LY2.append(N2)
									
			##### Conduct F-test of variances #####	    			
    			dfX = len(CFSR_stemp2.values) - 1
    			dfY = len(CFSR2_stemp2.values) - 1
    			F = CFSRvar/CFSR2var
    			#print(dfX,dfY,F)			
    			p_val2 = scipy.stats.f.cdf(F,dfX,dfY)
    			F_LY2.append(F)
    			P2_LY2.append(p_val2)			
    			site_LY2.append(i_string)
    			CFSR_mean_LY2.append(CFSRmean)
    			CFSR_var_LY2.append(CFSRvar)
    			CFSR2_mean_LY2.append(CFSR2mean)
    			CFSR2_var_LY2.append(CFSR2var)

    	elif(j == 2):
    		N1 = len(CFSR_stemp2.values)
    		N2 = len(CFSR_stemp2.values)

    		if (CFSR_check == False and CFSR2_check == False): #if both CFSR and CFSR2 have data
					
			##### Conduct t-test of means ######  
    			s = np.sqrt((CFSRvar/N1)+(CFSR2var/N2))
    			t = (CFSRmean - CFSR2mean)/s   #Welch's t-statistic
    			df = ((CFSRvar/N1)+(CFSR2var/N2))**2/((CFSRvar/(N1-1))**2+(CFSR2var/(N2-1))**2)   #d.f. for unequal sample size
    			t_score, p_value = stats.ttest_ind_from_stats(mean1=CFSRmean, std1=CFSRs, nobs1=N1, mean2=CFSR2mean, std2=CFSR2s, nobs2=N2, equal_var=False)
    			#print(t_score) 
    			T_LY3.append(t_score)
    			P1_LY3.append(p_value)
    			N1_LY3.append(N1)
    			N2_LY3.append(N2)
									
			##### Conduct F-test of variances #####	    			
    			dfX = len(CFSR_stemp2.values) - 1
    			dfY = len(CFSR2_stemp2.values) - 1
    			F = CFSRvar/CFSR2var			
    			#print(dfX,dfY,F)    			
    			p_val2 = scipy.stats.f.cdf(F,dfX,dfY)
    			F_LY3.append(F)
    			P2_LY3.append(p_val2)			
    			site_LY3.append(i_string)
    			CFSR_mean_LY3.append(CFSRmean)
    			CFSR_var_LY3.append(CFSRvar)
    			CFSR2_mean_LY3.append(CFSR2mean)
    			CFSR2_var_LY3.append(CFSR2var)

    	elif(j == 3):
    		N1 = len(CFSR_stemp2.values)
    		N2 = len(CFSR_stemp2.values)

    		if (CFSR_check == False and CFSR2_check == False): #if both CFSR and CFSR2 have data
					
			##### Conduct t-test of means ######  
    			s = np.sqrt((CFSRvar/N1)+(CFSR2var/N2))
    			t = (CFSRmean - CFSR2mean)/s   #Welch's t-statistic
    			df = ((CFSRvar/N1)+(CFSR2var/N2))**2/((CFSRvar/(N1-1))**2+(CFSR2var/(N2-1))**2)   #d.f. for unequal sample size
    			t_score, p_value = stats.ttest_ind_from_stats(mean1=CFSRmean, std1=CFSRs, nobs1=N1, mean2=CFSR2mean, std2=CFSR2s, nobs2=N2, equal_var=False)
    			#print(t_score) 
    			T_LY4.append(t_score)
    			P1_LY4.append(p_value)
    			N1_LY4.append(N1)
    			N2_LY4.append(N2)
									
			##### Conduct F-test of variances #####	    			
    			dfX = len(CFSR_stemp2.values) - 1
    			dfY = len(CFSR2_stemp2.values) - 1
    			F = CFSRvar/CFSR2var
    			#print(dfX,dfY,F)			
    			p_val2 = scipy.stats.f.cdf(F,dfX,dfY)
    			F_LY4.append(F)
    			P2_LY4.append(p_val2)			
    			site_LY4.append(i_string)
    			CFSR_mean_LY4.append(CFSRmean)
    			CFSR_var_LY4.append(CFSRvar)
    			CFSR2_mean_LY4.append(CFSR2mean)
    			CFSR2_var_LY4.append(CFSR2var)



stat_list_LY1 = {'Site':site_LY1,'CFSR Mean':CFSR_mean_LY1,'CFSR2 Mean':CFSR2_mean_LY1,'T-Stat':T_LY1,'P of T-Test':P1_LY1,'CFSR Variance':CFSR_var_LY1,'CFSR2 Variance':CFSR2_var_LY1,'F-Stat':F_LY1,'P of F-Test':P2_LY1}
dframe_LY1 = pd.DataFrame(stat_list_LY1)
#print(dframe_LY1)
dframe_LY1.to_csv('/mnt/data/users/herringtont/soil_temp/CFSR_stats/CFSR_stats_L1_deseasonalizedV2.csv',index=False)

stat_list_LY2 = {'Site':site_LY2,'CFSR Mean':CFSR_mean_LY2,'CFSR2 Mean':CFSR2_mean_LY2,'T-Stat':T_LY2,'P of T-Test':P1_LY2,'CFSR Variance':CFSR_var_LY2,'CFSR2 Variance':CFSR2_var_LY2,'F-Stat':F_LY2,'P of F-Test':P2_LY2}
dframe_LY2 = pd.DataFrame(stat_list_LY2)
dframe_LY2.to_csv('/mnt/data/users/herringtont/soil_temp/CFSR_stats/CFSR_stats_L2_deseasonalizedV2.csv',index=False)

stat_list_LY3 = {'Site':site_LY3,'CFSR Mean':CFSR_mean_LY3,'CFSR2 Mean':CFSR2_mean_LY3,'T-Stat':T_LY3,'P of T-Test':P1_LY3,'CFSR Variance':CFSR_var_LY3,'CFSR2 Variance':CFSR2_var_LY3,'F-Stat':F_LY3,'P of F-Test':P2_LY3}
dframe_LY3 = pd.DataFrame(stat_list_LY3)
dframe_LY3.to_csv('/mnt/data/users/herringtont/soil_temp/CFSR_stats/CFSR_stats_L3_deseasonalizedV2.csv',index=False)

stat_list_LY4 = {'Site':site_LY4,'CFSR Mean':CFSR_mean_LY4,'CFSR2 Mean':CFSR2_mean_LY4,'T-Stat':T_LY4,'P of T-Test':P1_LY4,'CFSR Variance':CFSR_var_LY4,'CFSR2 Variance':CFSR2_var_LY4,'F-Stat':F_LY4,'P of F-Test':P2_LY4}
dframe_LY4 = pd.DataFrame(stat_list_LY4)
dframe_LY4.to_csv('/mnt/data/users/herringtont/soil_temp/CFSR_stats/CFSR_stats_L4_deseasonalizedV2.csv',index=False)

