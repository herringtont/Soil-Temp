####### Difference Method #########
#y_rescale_factor = np.mean((x-x_bar)*(z-z_bar))/np.mean((y-y_bar)*(z-z_bar))
#z_rescale_factor = np.mean((x-x_bar)*(y-y_bar))/np.mean((z-z_bar)*(y-y_bar)) 

#y_rescaled = y_rescale_factor*(y-y_bar)+x_bar
#z_rescaled = z_rescale_factor(z-z_bar)+x_bar

#err_var_x = np.mean((x-y_rescaled)*(x-z_rescaled))
#err_var_y = np.mean((y_rescaled-x)*(y_rescaled - z_rescaled))
#err_var_z = np.mean((z_rescaled-x)*(z_rescaled - y_rescaled))

## Note that Difference Method estimates error variances within the dataspace of the reference dataset, whereas error variances in the covariance formulation are estimated in their own dataspaces

######## Covariance Method #############


### signal variances
#signal_var_x = cov(x,y)*cov(x,z)/cov(y,z)
#signal_var_y = cov(y,x)*cov(y,z)/cov(x,z)
#signal_var_z = cov(z,x)*cov(z,y)/cov(x,y)

### rescaling parameters
#beta_star_y = cov(x,z)/cov(y,z)
#beta_star_z = cov(x,y)/cov(z,y)

### error variances
#err_var_x = var(x) - signal_var_x
#err_var_y = var(y) - signal_var_y
#err_var_z = var(z) - signal_var_z

### Signal-to-Noise Ratio (SNR)
#SNRx = signal_var_x/err_var_x
#SNRy = signal_var_y/err_var_y
#SNRz = signal_var_z/err_var_z

### Noise-to-Signal Ratio (NSR)
#NSRx = err_var_x/signal_var_x
#NSRy = err_var_y/signal_var_y
#NSRz = err_var_z/signal_var_z

### rescaled data
#y_rescaled = (beta_star_y*(y-y_bar))+x_bar
#z_rescaled = (beta_star_z*(z-z_bar))+x_bar

### RMSD
#RMSDxy = np.mean((x-y)**2))**0.5
#RMSDyz = np.mean((y-z)**2)**0.5
#RMSDxz = np.mean((x-z)**2)**0.5

###ubRMSE

#ubRMSEx = math.sqrt(abs(var(x) - signal_var_x))
#ubRMSEy = math.sqrt(abs(var(y) - signal_var_y))
#ubRMSEz = math.sqrt(abs(var(z) - signal_var_z))

##Note that if the ubRMSE is calculated using the difference method, 

### fRMSE
#fRMSEx = err_var_x/(signal_var_x+err_var_x)
#fRMSEy = err_var_y/(signal_var_y+err_var_y)
#fRMSEz = err_var_z/(signal_var_z+err_var_z)

### Cross-Correlations (assuming error-orthoginality and error cross-correlations to be negligible, as is done in TC)
#Rxy = 1/((1+NSRx)*(1+NSRy)) 
#Ryz = 1/((1+NSRy)*(1+NSRz))
#Rxz = 1/((1+NSRx)*(1+NSRz)) 

### Correlation of dataset with true signal
#Rx = signal_var_x/(signal_var_x+err_var_x)
#Ry = signal_var_y/(signal_var_y+err_var_y)
#Rz = signal_var_z/(signal_var_z+err_var_z)

