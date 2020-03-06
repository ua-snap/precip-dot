# generate test data for WebApp

# add spatial WRF attributes
def wrf_attr(ds):
    ds.attrs['proj_parameters'] = era.attrs['proj_parameters']
    ds.attrs['crs_wkt'] = era.attrs['crs_wkt']
    return ds

# make array of mean "scalings" between 1hr, 1yr PF value and all others
def mean_scale(directory):
    fns = os.listdir(directory)
    fps = [os.path.join(directory, fn) for fn in fns]
    arrs = []
    for fp in fps:
        d = pd.read_csv(fp, skiprows=14, nrows=15, header=None)
        d = d.drop(0, axis=1)
        # create scaling factors by dividing by PF for 1hr, 1yr
        arrs.append(np.round(d.values / d.values[3][0], 2))
    arr = np.stack(arrs, axis=0)
    return arr.mean(axis=0)



if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    import xarray as xr
    import os

    # read example precip data file for inspiration
    print('Reading sample ERA-Interim precip output')
    era = xr.open_dataset('pcpt_hourly_wrf_ERA-Interim_historical_1992.nc').load()
    print('Done.', '\n')

    # make scaling factors from sample PF estimates    
    sf = mean_scale("sample_pf_tables")

    # use max values as baseline to scale, corresponding to (1hr, 1yr) PF bin
    # convert to inches
    print('Calculating baseline 1hr/1yr values')
    base_ds = era.max(dim='time')/25.4
    std_ds = era.std(dim='time')/25.4
    print('Done.', '\n')

    # durations and RIs to iterate through
    dur = ["5min", "10min", "15min", "30min", "60min", "2hr", "3hr", "6hr", 
           "12hr", "24hr", "2d", "3d", "4d", "7d", "10d", "20d", "30d", "45d", "60d"]
    ri = ["1y", "2y", "5y", "10y", "25y", "50y", "100y", "200y", "500y", "1000y"]

    # create test_data/ dir if not exist
    if not os.path.exists("test_data"):
        os.mkdir("test_data")
    # iterate through integers for saving with duration suffix
    print('Generating data...')
    for i in range(15):
        ds = []
        for j in range(10):
            ds.append(base_ds * sf[i, j])
        # combine datasets creating new axis
        mean_ds = xr.concat(ds, 'ri')
        mean_ds.coords['ri'] = ri
        mean_ds = wrf_attr(mean_ds)
        ucl_ds = mean_ds + (std_ds * 2)
        lcl_ds = mean_ds - (std_ds * 2)
        # save mean, upper and lower confidence limits
        mean_ds.to_netcdf('test_data/{}_mean_pf_test.nc'.format(dur[i]))
        ucl_ds.to_netcdf('test_data/{}_ucl_pf_test.nc'.format(dur[i]))
        lcl_ds.to_netcdf('test_data/{}_lcl_pf_test.nc'.format(dur[i]))
    print('Done. Data saved to "test_data/"')
