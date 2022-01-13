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
from scipy.stats import pearsonr
from matplotlib.ticker import (MultipleLocator, AutoLocator, AutoMinorLocator)

##### Define Functions #####

def scatter_subset(x, y, hue, mask, **kws):
    sns.scatterplot(x=x[mask], y=y[mask], hue=hue[mask], **kws)


##### Grab Data ######
seas_cycle_dir = '/mnt/data/users/herringtont/soil_temp/seasonal_cycle/'
top_NAm_fil = ''.join([seas_cycle_dir+'remapcon_simple_average_zscore_top_30cm_thr100_seasonal_cycle_NAm.csv'])
top_Eur_fil = ''.join([seas_cycle_dir+'remapcon_simple_average_zscore_top_30cm_thr100_seasonal_cycle_Eur.csv'])
btm_NAm_fil = ''.join([seas_cycle_dir+'remapcon_simple_average_zscore_30cm_300cm_thr100_seasonal_cycle_NAm.csv'])
btm_Eur_fil = ''.join([seas_cycle_dir+'remapcon_simple_average_zscore_30cm_300cm_thr100_seasonal_cycle_Eur.csv'])


top_NAm_dframe = pd.read_csv(top_NAm_fil)
top_Eur_dframe = pd.read_csv(top_Eur_fil)
btm_NAm_dframe = pd.read_csv(btm_NAm_fil)
btm_Eur_dframe = pd.read_csv(btm_Eur_fil)

gcell_top_NAm = top_NAm_dframe['Grid Cell'].values
gcell_top_NAm_uq = np.unique(gcell_top_NAm)

gcell_top_Eur = top_Eur_dframe['Grid Cell'].values
gcell_top_Eur_uq = np.unique(gcell_top_Eur)

gcell_btm_NAm = btm_NAm_dframe['Grid Cell'].values
gcell_btm_NAm_uq = np.unique(gcell_btm_NAm)

gcell_btm_Eur = btm_Eur_dframe['Grid Cell'].values
gcell_btm_Eur_uq = np.unique(gcell_btm_Eur)
