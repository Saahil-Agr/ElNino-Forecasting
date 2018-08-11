from netCDF4 import Dataset
import numpy as np
from os.path import join
import pandas as pd

np.set_printoptions(suppress=True)



dataset = Dataset('/Volumes/matiascastilloHD/CLIMATEAI/GCMs/IRI_wind/real_scenario/raw/vwind_IRI.nc')

time_steps = dataset.variables['T'][:]



lat = dataset.variables['Y'][:]
print(lat)
lon = dataset.variables['X'][:]
print(lon)

for date in time_steps:
    f_interp = RegularGridInterpolator(
                                gcm_grid,
                                anoms,
                                bounds_error=False,
                                fill_value=None,
                            )


lat_li = np.asscalar(np.where(lat == 5)[0])
lat_ui = np.asscalar(np.where(lat == -5)[0])
lon_li = np.asscalar(np.where(lon == 190)[0])
lon_ui = np.asscalar(np.where(lon == 240)[0])

# 12 months of ENSO prediction for specific starting time.
ENSO = np.average(dataset.variables['ts'][:,0,lat_li:lat_ui+1,lon_li:lon_ui+1], axis=(1,2))
predictions[str(year)+'-'+month] = ENSO - 273.15


dataframe = pd.DataFrame.from_dict(predictions)
#dataframe.to_csv(join(folder,model+'_simulations.csv'))
