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

############################# set directories ###########################

naive_dir = "/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/Naive/raw_temp/"
naive_dir_anom = "/mnt/data/users/herringtont/soil_temp/Blended_Product/collocated/Naive/anom/"

olr = ['outliers','zscore','IQR']
lyr = ['0_9.9']
thr = ['0','25','50','75','100']
rmp_type = ['nn','bil']




