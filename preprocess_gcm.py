"""
script to preprocess GCM data for use in spatiotemporal image processing networks

file structure
    | `data_dir`
    |-- GCMs
    |---- `gcm_name`
    |------ `scenario`
    |-------- raw
    |-------- regrid_anomalies
    |-------- img

right now using bleeding edge version of xarray 
https://github.com/pydata/xarray/pull/1252
"""
from os.path import join, split

import xarray as xr
import pandas as pd 
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
from tqdm import tqdm
from PIL import Image
import dask


def main(preprocess_netcdfs=False, 
         generate_target=False, 
         write_images=True, 
         data_dir='.', 
         grid_nc='', 
         gcm_names=('CNRM-CM5',), 
         scenarios={'CNRM-CM5': ('piControl_r1i1p1',)}, 
         channels=('tas',), 
         target=None,
         img_ext='npy', 
         img_type=np.float32):
    """
    script with three gcm preprocessing functionalities

    preprocess_netcdfs: 
        bool, default False
        open netcdf files containing gcm output and interpolate onto chosen grid.
        remove seasonal mean climate from data to generate anomalies.
        save regridded anomalies by variable as a new set of netcdf files.

    generate_target:
        bool, default False
        open preprocessed netcdf files containing regridded anomalies.
        given a bounding box and variable compute a timeseries of the spatial mean
        at each timestep. 
        save as a .csv file that can be read into a pandas DataFrame

    write_images:
        bool, default True
        open preprocessed netcdf files containing regridded anomaly data.
        write each timestep to an individual image file 

    data_dir:
        string, default '.'
        path to top level data directory

    grid_nc:
        string, default '',
        path to netcdf file containing the target grid for interpolation,

    gcm_names:
        sequence, default ('CNRM-CM5',)
        sequence of strings specifying names of GCMs to preprocess

    scenarios:
        dict, default {'CNRM-CM5': ('piControl_r1i1p1',)}
        keys are string names of GCMs, values are sequences of strings 
        specifying scenarios for this gcm

    channels:
        sequence, default ('tas',)
        sequence of strings giving variables in climate model to analyze
        the order of the variables in this sequence will be the order 
        of the channels in the image files 

    target:
        None or ClimateTarget instance, default None
        ClimateTarget instance specifying parameters for calculating 
        response time series

    img_ext:
        string, default 'npy'
        'npy' to save as serialized numpy arrays
        'jpg' to save as jpeg image files

    img_type:
        data type, default np.float32
        type to save data in output img files. 
        must be np.uint8 to save PIL jpgs
        if np.uint8 or np.uint32 will trigger scaling of raw data
    """
    ## needed until xarray 0.11 is released
    xr.set_options(enable_cftimeindex=True)
    ## interpolate onto common grid and calculate anomalies
    if preprocess_netcdfs:
        print("Preprocessing NetCDF Data")
        with xr.open_dataset(join(data_dir, grid_nc)) as grid_ds:
            ## interpolate gcm onto this grid
            Y, X = np.meshgrid(grid_ds.lat, grid_ds.lon)
            xi = np.asarray([Y.flatten(), X.flatten()]).T
            ## name for regridded files
            rg_fstr = '{}_Amon_{}_{}_{}_{}_%sx%s_anomalies.nc' % (str(X.shape[0]), str(X.shape[1]))
            for gcm_name in gcm_names:
                for scenario in scenarios[gcm_name]:
                    for k, var in enumerate(channels):
                        print(var, gcm_name, scenario)
                        ## path for gcm files
                        gcm_path = join(data_dir, 'GCMs', gcm_name, scenario)
                        ## 
                        gcm = xr.open_mfdataset(join(gcm_path, 'raw', '{}*.nc'.format(var)))
                        ## only need to do this once
                        if k == 0:
                            date_strings = [str(date)[:7] for date in gcm['time'].values]
                            dr = pd.PeriodIndex(date_strings, freq='M')
                            ## for outer loop over 100 years at a time
                            n_periods = np.ceil((dr[-1].year - dr[0].year) / 100)
                            year_range = pd.period_range(start=str(dr[0])[:7], periods=n_periods, freq='100A')
                            gcm_grid = (np.asarray(gcm.lat), np.asarray(gcm.lon))
                        ## preallocate array to store mean values
                        ss = np.zeros(gcm[var][:12,:,:].shape, dtype=np.float64)
                        ## compute anomalies by removing mean for each location for each month
                        for m in range(1, 13):
                            print('Computing anomalies for month', m)
                            ## compute the mean value for each month
                            ss[m-1,:,:] = gcm[var][dr.month == m,:,:].mean(dim='time')
                        ## iterate in 100 year chunks to keep new .nc file sizes under control
                        for y0, y1 in zip(year_range[:-1], year_range[1:]):
                            print('\nInterpolating', var, y0, y1)
                            filtr = (dr.year >= y0.year) & (dr.year < y1.year)
                            ## number of timesteps
                            n_t = np.sum(filtr)
                            ## output data will be stored here
                            sub_gcm = xr.DataArray(
                                np.zeros((n_t, grid_ds.lat.shape[0], grid_ds.lon.shape[0])),
                                coords=[
                                    ('time', gcm.time[filtr]), 
                                    ('lat', grid_ds.lat),
                                    ('lon', grid_ds.lon)
                                ]
                            )
                            sub_gcm.attrs = gcm[var].attrs
                            i = np.where(filtr)[0][0]
                            ## subloop over each month in the 100 year interval
                            for j in tqdm(range(n_t)):
                                m = dr[i+j].month
                                anoms = np.asarray(gcm[var][i+j,:,:]) - ss[m-1,:,:]
                                ## interpolate gcm to ncar grid
                                f_interp = RegularGridInterpolator(
                                    gcm_grid, 
                                    anoms,
                                    # bounds_error=False,
                                    # fill_value=None,
                                )
                                interp_data = f_interp(xi, method='linear')
                                sub_gcm[j,:,:] = interp_data.reshape(Y.shape).T
                            sub_gcm = sub_gcm.to_dataset(name=var)
                            sub_gcm.attrs = gcm.attrs
                            ## path to write out file
                            fname = rg_fstr.format(var, gcm_name, scenario, str(y0)[:7], str(y1)[:7])
                            fp = join(gcm_path, 'regrid_anomalies', fname)
                            ## save 100 years of regridded anomalies data as .nc file
                            sub_gcm.to_netcdf(fp, mode='w', unlimited_dims=['time'], engine='netcdf4')
                        gcm.close()
        print('\nFinished Interpolating')
    ## calculate a timeseries to use as a response variable
    if generate_target:
        print('\nCalculating', target.name)
        ## loop over GCMs and scenarios
        for gcm_name in gcm_names:
            for scenario in scenarios[gcm_name]:
                print(gcm_name, scenario)
                # gcm_path = join(data_dir, 'GCMs', gcm_name, scenario, '*.nc')
                gcm_path = join(data_dir, 'GCMs', gcm_name, scenario)
                with xr.open_mfdataset(join(gcm_path, 'regrid_anomalies', '*.nc')) as gcm:    
                    ## get spatial subset for variable
                    gcm_sub = gcm[target.var].sel(lon=slice(*target.lon_range), lat=slice(*target.lat_range))
                    ## calculate spatial mean
                    y = gcm_sub.mean(dim=['lon', 'lat'])
                    fname = '{}_{}_{}.csv'.format(gcm_name, scenario, target.name)
                    fp = join(gcm_path, fname)
                    y.to_dataframe().to_csv(fp)
        print('\nFinished computing {}'.format(target.name))
    ## Save each month's data as an individual file
    if write_images:
        print('\nWriting {} files'.format(img_ext))
        channels_str = (len(channels) * '{}-').format(*channels)[:-1]
        img_fstr = '%s_Amon_{}_{}_{}.%s' % (channels_str, img_ext)
        ## number of bits depending on datatype
        if img_type is np.uint32:
            bits = 2 ** 32 - 1
        elif img_type is np.uint8:
            bits = 2 ** 8 - 1
        ## loop over GCMs and scenarios
        for gcm_name in gcm_names:
            for scenario in scenarios[gcm_name]:
                print(gcm_name, scenario)
                # gcm_path = join(data_dir, 'GCMs', gcm_name, scenario, '*.nc')
                gcm_path = join(data_dir, 'GCMs', gcm_name, scenario)
                with xr.open_mfdataset(join(gcm_path, 'regrid_anomalies', '*.nc')) as gcm:
                    ## if using an integer type, scale to full range of integer
                    if img_type in (np.uint8, np.uint32):
                        print('\nScaling channels')
                        for var in channels:
                            print('  {}'.format(var))
                            vmin = gcm[var].min()
                            vmax = gcm[var].max()
                            ## normalize entire array to 0-1 and scale to image bits (ie 0-255 for 8 bit)
                            gcm[var] = bits * (gcm[var] - vmin) / vmax
                    ## write image file at each timestep
                    print('\nGenerating images for each timestep')
                    for i, t in tqdm(enumerate(gcm.time)):
                        ## for each channel load data for timestep t into memory and flip array upright
                        vals  = [np.flipud(np.asarray(gcm[var].isel(time=i))) for var in channels]
                        ## stack channels together and specify data type, shape = (rows, cols, channels)
                        arr = np.stack(vals, axis=-1).astype(img_type)
                        name = img_fstr.format(gcm_name, scenario, str(t.values)[:7])
                        img_path = join(gcm_path, 'img', name)
                        ## .npy file format can take arbitrary color bands and pixel bits
                        if img_ext == 'npy':
                            np.save(img_path, arr)
                        ## .jpg with pillow can take 1, 3, or 4 color bands and 8 bits
                        elif img_ext == 'jpg' and len(channels) == 3:
                            Image.fromarray(arr, mode='RGB').save(img_path)
                        elif img_ext == 'jpg' and len(channels) == 1:
                            Image.fromarray(arr, mode='L').save(img_path)
        print('\nFinished writing {} files'.format(img_ext))


class ClimateTarget:
    """data structure for climate variable bounding box"""
    def __init__(self, name, var, lat_range, lon_range):
        """for lon_range be careful of crossing the date line """
        self.name = name
        self.var = var 
        self.lat_range = lat_range
        self.lon_range = lon_range


if __name__ == '__main__':
    ##############
    ## SETTINGS ##
    ##############
    ## path to the .nc file whose grid we will interpolate onto (the GCM with the coarsest grid)
    grid_nc = 'GCMs/MPI-ESM-LR/piControl_r1i1p1/raw/tas_Amon_MPI-ESM-LR_piControl_r1i1p1_185001-203512.nc'
    ## these are the GCMs to preprocess
    gcm_names = ['CNRM-CM5']#, 'MPI-ESM-LR', 'CanESM2', 'HadGEM2-ES',]
    ## these are the experiments and forcings settings for each gcm
    scenarios = {
        'CNRM-CM5': ('piControl_r1i1p1',),
    }
    ## these are the variables to work with
    channels = ['tas', 'uas', 'psl']
    ## variable, latitude range, and longitude range for the target 
    ## Nino3.4 index 
    ## https://www.climate.gov/news-features/blogs/enso/why-are-there-so-many-enso-indexes-instead-just-one
    target = ClimateTarget(name='NINO_3-4', var='tas', lat_range=[-5, 5], lon_range=[190, 240])
    ## precipitation over India
    # target = ClimateTarget(name='India_precip', var='pr', lat_range=[10, 25], lon_range=[70, 90])
    main(
        preprocess_netcdfs=True,
        generate_target=True, 
        write_images=True, 
        data_dir='/Volumes/CLIMATEAI', 
        grid_nc=grid_nc,
        gcm_names=gcm_names,
        scenarios=scenarios,
        channels=channels,
        target=target,
        img_ext='jpg'
    )

