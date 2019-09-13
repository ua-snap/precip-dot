# NOTES:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html
# pip install lmoments3
# https://lmoments3.readthedocs.io/en/stable/#l-moment-estimation-from-sample-data
# THE ONE BELOW MIGHT BE REALLY IMPORTANT:
# https://www.linkedin.com/pulse/beginners-guide-carry-out-extreme-value-analysis-2-chonghua-yin/
# https://github.com/royalosyin/A-Beginner-Guide-to-Carry-out-Extreme-Value-Analysis-with-Codes-in-Python

def return_intervals(dat, avi):
    # get LMOMENTS
    # LMU = lmom.lmom_ratios(dat)
    paras = distr.gev.lmom_fit(dat)
    fitted_gev = distr.gev(**paras)

    pavi = np.empty(len(avi))
    for i in range(len(avi)):
        pavi[i] = 1.0-1.0 / avi[i]
    
    # return year precip estimates
    return fitted_gev.ppf(pavi)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    import xarray as xr
    import lmoments3 as lmoments
    from lmoments3 import distr
    # import lmoments
    import pandas as pd
    import geopandas as gpd
    import rasterio
    import numpy as np

    # read in some point locations to use in testing...
    # {'Anchorage', 'Fairbanks', 'Barrow', 'Nome', 'Dillingham', 'Juneau', 'Seward'}
    # shp = gpd.read_file('/workspace/UA/malindgren/repos/precip-dot/ancillary/alaska_test_locations.shp')

    # read in PCPT from wrf
    # fn = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/PCPT_wrf_hourly_gfdl_rcp85_2020_1D_ams.nc'
    # files = sorted([i for i in glob.glob('/workspace/Shared/Tech_Projects/wrf_data/project_data/wrf_data/daily/pcpt/pcpt_hourly_wrf_GFDL-CM3_rcp85_*.nc') if '201' in i or '202' in i or '203' in i or '204' in i])
    files = '/workspace/Shared/Tech_Projects/wrf_data/project_data/wrf_data/daily/pcpt/pcpt_*_wrf_GFDL-CM3_historical_*.nc'
    ds = xr.open_mfdataset( files )
    ds = ds.sel( time=slice('1970','1990') ).load()

    # # some constant metadata about the WRF grid
    # res = 20000 # 20km resolution
    # x0,y0 = np.array( ds.xc.min()-(res/2.)), np.array(ds.yc.max()+(res/2.) ) # origin point upper-left corner from centroid
    # wrf_crs = '+units=m +proj=stere +lat_ts=64.0 +lon_0=-152.0 +lat_0=90.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000'
    # # make an affine transform
    # a = rasterio.transform.from_origin( x0, y0, res, res ) # build affine transform using rasterio mechanics
    # shp_wrf = shp.to_crs(wrf_crs)
    # lonlat = shp_wrf.geometry.apply(lambda x: (x.x, x.y)).tolist()

    # ds_day = ds.resample(time='1D').mean().compute()
    # # write out the daily average?
    # ds_day.to_netcdf('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/pcpt_daily_wrf_GFDL-CM3_historical.nc')

    # # grab daily profiles for some locations
    # rowcols = [ ~a*lola for lola in lonlat ]
    # rowcols = {name:(int(rc[0]), int(rc[1]))for name, rc in zip(shp_wrf.NameX.tolist(),rowcols)}
    # profiles = { name:ds.pcpt[:,rc[0],rc[1]].values for name, rc in rowcols.items() }
    # data = pd.DataFrame(profiles, index=ds.time.to_index())
    # data = data[~pd.isna(data)] # remove the errant missing data...  THIS NEEDS FIXING PROJECT-WIDE
    # # make a data frame -- to mimick the tutorial
    # data['YYYY'] = [ i.year for i in data.index ]
    # data['MM'] = [ i.month for i in data.index ]
    # data['DD'] = [ i.day for i in data.index ]
    # # TEST
    # data = pd.read_csv('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/pcpt_extracted_gfdl_historical_daily.csv',index_col=0)
    # # end TEST
    # df = data[['YYYY','MM','DD','Anchorage']]
    # df.columns = ['YYYY','MM','DD','PRCP']

    # # Select annual maxima as extreme values
    # df = df.groupby("YYYY").max()
    # dat = df.PRCP[~np.isnan(df.PRCP)]

    # set some return years
    avi  = [2,5,10,20,50,100,200,500,1000];

    arr = ds.pcpt.values
    # loop through a 3D array and make a new 3D array with the returns
    rows=262
    cols=262
    out = np.zeros((len(avi), rows, cols ), dtype=np.float32)
    for i,j in np.ndindex(arr.shape[-2:]):
        out[:,i,j] = return_intervals( arr[:,i,j], avi )

    # now from here we can put those into a new NetCDF file
    
