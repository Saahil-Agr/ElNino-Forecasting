"""
climate indices from http://etccdi.pacificclimate.org/list_27_indices.shtml
"""
from os.path import join

import xarray as xr
from xarray.ufuncs import isfinite
import pandas as pd 
from matplotlib import pyplot as plt 
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
from shapely.geometry import Point, mapping
from fiona import collection
from statsmodels.distributions.empirical_distribution import ECDF
import dask
from dask.distributed import Client
from PIL import Image
# client = Client()


def map_360_to_365_day_calendar(date_strings):
    """
    given dates on a 360 day calendar w/ 30 day months, map to real 365 day calendar
    :param date_strings:, vector of strings in form "YYYY-MM-DD"
    :return: pandas.DatetimeIndex
    """
    dt = pd.to_datetime(date_strings, errors='coerce')
    ## TODO handle edge case where first or last date is invalid?
    start, end = dt[0], dt[-1]
    dr = pd.date_range(start=start, end=end)
    ## filter the 29th on leap years
    f1 = (dr.month == 2) & (dr.day == 29)
    ## filter 5 31st on other years
    f2 = (dr.day == 31) & ((dr.month == 1) | (dr.month == 3) | (dr.month == 7) \
                                          | (dr.month == 10) | (dr.month == 12))
    ## remove from date range
    dr = dr.delete(np.where(f1 | f2)[0])
    ## for most cases the function passes this assertion
    try:
        assert len(date_strings) == len(dr)
    ## this UGLINESS is to catch missing months of data
    ## eg HadGEM2-ES_historical dataset is missing Nov, 2000
    except AssertionError:
        bad_months = {}
        dt = dt[np.logical_not(np.isnan(dt.year))]
        ## check if years match
        for yr in set(dt.year):
            delta =  np.nansum(dt.year == yr) - np.nansum(dr.year == yr)
            leap = pd.to_datetime('{:d}-01-01'.format(yr)).is_leap_year
            if delta != 5 + leap:
                bad_months[yr] = []
        for yr in bad_months.keys():
            ## check if months match
            for mth in range(1, 13):
                dt_sum = np.nansum((dt.year == yr) & (dt.month == mth))
                dr_sum = np.nansum((dr.year == yr) & (dr.month == mth))
                if mth is 2 and pd.to_datetime('{:d}-01-01'.format(yr)).is_leap_year:
                    bump = -1
                elif mth in {5, 8}:
                    bump = 1
                else:
                    bump = 0
                if dt_sum + bump != dr_sum and dr_sum:
                    bad_months[yr].append(mth)
                    print("Expected", dt_sum+bump, "Found", dr_sum)
                    print("Removing {:d}/{:d}".format(mth, yr))
        for yr, m_list in bad_months.items():
            for mth in m_list:
                inds = np.where((dr.year == yr) & (dr.month == mth))[0]
                dr = dr.delete(inds)
        assert len(date_strings) == len(dr)
    return dr


def main(data_dir='.', preprocess_netcdfs=False, write_images=True, 
         generate_target=False, img_ext='npy', img_type=np.uint32):
    """
    script with three gcm preprocessing functionalities

    preprocess_netcdfs: 
        open netcdf files containing gcm output and interpolate onto chosen grid.
        remove seasonal mean climate from data to generate anomalies.
        save regridded anomalies by variable as a new set of netcdf files.

    generate_target:
        open preprocessed netcdf files containing regridded anomalies.
        given a bounding box and variable compute a timeseries of the spatial mean
        at each timestep. 
        save as a .csv file that can be read into a pandas DataFrame

    write_images:
        open preprocessed netcdf files contain
    """
    ## SETTINGS
    grid_nc = 'GCMs/MPI-ESM-LR/piControl/tas_Amon_MPI-ESM-LR_piControl_r1i1p1_185001-203512.nc'
    gcm_names = ['CNRM-CM5']#, 'CanESM2', 'HadGEM2-ES', 'MPI-ESM-LR',]
    scenarios = {
        'CNRM-CM5': ['piControl_r1i1p1'],
    }
    channels = ['tas', 'psl']
    if preprocess_netcdfs:
        with xr.open_dataset(join(data_dir, grid_nc)) as grid_ds:
            ## interpolate gcm onto this grid
            Y, X = np.meshgrid(grid_ds.lat, grid_ds.lon)
            xi = np.asarray([Y.flatten(), X.flatten()]).T
            ## name for regridded files
            rg_fstr = '{}_Amon_{}_{}_{}_{}_%sx%sregrid_anomalies.nc' % (str(X.shape[0]), str(X.shape[1]))
            for gcm_name in gcm_names:
                for scenario in scenarios[gcm_name]:
                    gcm_path = join(data_dir, 'GCMs', gcm_name, scenario, '*.nc')
                    with xr.open_mfdataset(gcm_path) as gcm:
                        gcm_grid = (np.asarray(gcm.lat), np.asarray(gcm.lon))
                        # gcm to 365 day calendar
                        date_strings = [str(d)[:10] for d in gcm.time.values]
                        try:
                            dr = pd.to_datetime(date_strings)
                        except ValueError:
                            dr = map_360_to_365_day_calendar(date_strings)
                        gcm['time'] = dr 
                        ## loop over years
                        year_range = pd.date_range(dr[0], dr[-1], frequency='100Y')
                        for var in channels:
                            ## compute anomalies by removing seasonal mean for dataset
                            for m in range(1, 13):
                                filtr = dr.month == m
                                ss = gcm[var][filtr,:,:].mean()
                                gcm[var][filtr,:,:] = gcm[var][filtr,:,:] - ss
                            for y0, y1 in zip(year_range[:-1], year_range[1:]):
                                print('Interpolating', var, y0, y1)
                                filtr = (dr.year >= y0) and (dr.year < y1)
                                dt = np.sum(filtr)
                                sub_gcm = xr.DataArray(
                                    np.zeros((dt, ncar.lat.shape[0], ncar.lon.shape[0])),
                                    coords=[
                                        ('time', gcm.time[filtr]), 
                                        ('lat', ncar.lat),
                                        ('lon', ncar.lon)
                                    ]
                                )
                                sub_gcm.attrs = gcm[var].attrs
                                ## loop over days in that year
                                i = np.min(np.where(filtr)[0])
                                for j in range(dt):
                                    ## interpolate gcm to ncar grid
                                    f_interp = RegularGridInterpolator(
                                        gcm_grid, 
                                        np.asarray(gcm[var][i + j,:,:]),
                                        bounds_error=False,
                                        fill_value=None,
                                    )
                                    interp_data = f_interp(xi, method='linear')
                                    sub_gcm[j,:,:] = interp_data.reshape(Y.shape).T
                                sub_gcm = sub_gcm.to_dataset(name=var)
                                sub_gcm.attrs = gcm.attrs
                                fp = join(gcm_path, rg_fstr.format(var, gcm_name, scenario, y0, y1))
                                sub_gcm.to_netcdf(fp, mode='w', unlimited_dims=['time'])
        print('Finished Interpolating')
    if generate_target:
        pass
    if write_images:
        print('Writing image files')
        channels_str = (len(channels) * '{}-').format(*channels)[:-1]
        img_fstr = '%s_Amon_{}_{}_{}-{}.%s' % (channels_str, img_ext)
        if img_type is np.uint32:
            bits = 2 ** 32 - 1
        elif img_type is np.uint8:
            bits = 2 ** 8 - 1
        for gcm_name in gcm_names:
            for scenario in scenarios[gcm_name]:
                gcm_path = join(data_dir, 'GCMs', gcm_name, scenario, '*.nc')
# gcm_path = join(data_dir, 'GCMs', gcm_name, scenario, '*regrid_anomalies.nc')
                with xr.open_mfdataset(gcm_path) as gcm:
                    for var in channels:
                        vmin = gcm[var].min()
                        vmax = gcm[var].max()
                        ## normalize entire array to 0-1 and scale to image bits
                        gcm[var] = scale * (gcm[var] - vmin) / vmax
                    ## write image file at each timestep
                    for t in gcm.time:
                        arr = np.asarray(
                            [gcm[var].sel(time=t) for var in channels],
                            dtype=img_type
                        )
                        name = img_fstr.format(gcm_name, scenario, t.year, t.month)
                        path = join(data_dir, 'img', gcm, scenario, name)
                        if img_ext == 'npy':
                            np.save(arr, path)
                        elif image_ext == 'jpg':
                            # Image.fromarray(arr).save(path)
                            pass




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('error')
    main(data_dir='.')

