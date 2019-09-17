# NOTES:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html
# pip install lmoments3
# https://lmoments3.readthedocs.io/en/stable/#l-moment-estimation-from-sample-data
# THE ONE BELOW MIGHT BE REALLY IMPORTANT:
# https://www.linkedin.com/pulse/beginners-guide-carry-out-extreme-value-analysis-2-chonghua-yin/
# https://github.com/royalosyin/A-Beginner-Guide-to-Carry-out-Extreme-Value-Analysis-with-Codes-in-Python

def return_intervals(dat, avi):
    if not (dat == 0).all():
        # get LMOMENTS
        # LMU = lmom.lmom_ratios(dat)
        paras = distr.gev.lmom_fit(dat)
        fitted_gev = distr.gev(**paras)

        pavi = np.empty(len(avi))
        for i in range(len(avi)):
            pavi[i] = 1.0-1.0 / avi[i]
        
        # return year precip estimates
        return fitted_gev.ppf(pavi)
    else:
        return np.repeat(np.nan, len(avi))

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    import xarray as xr
    import lmoments3 as lmom
    from lmoments3 import distr
    import pandas as pd
    import geopandas as gpd
    import rasterio, itertools
    import numpy as np
    import glob, os

    # read in some point locations to use in testing...
    path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt'
    data_groups = ['ERA-Interim_historical','GFDL-CM3_historical','GFDL-CM3_rcp85']
    durations = ['10D','12H','1D','1H','20D','2D','2H','30D','3D','3H','45D','4D','60D','6H','7D']

    for data_group, duration in itertools.product(data_groups, durations):

        # read in PCPT from wrf
        files = sorted(glob.glob(os.path.join(path,'pcpt_*_{}_*{}*.nc'.format(data_group,duration))))
        ds = xr.open_mfdataset( files ).load()

        # some constant metadata about the WRF grid
        res = 20000 # 20km resolution
        x0,y0 = np.array( ds.xc.min()-(res/2.)), np.array(ds.yc.max()+(res/2.) ) # origin point upper-left corner from centroid
        wrf_crs = '+units=m +proj=stere +lat_ts=64.0 +lon_0=-152.0 +lat_0=90.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000'
        # make an affine transform
        a = rasterio.transform.from_origin( x0, y0, res, res ) # build affine transform using rasterio mechanics

        # calc ann_max
        ds_ann_max = ds.pcpt.resample(time='y').max(axis=0)
        
        # set some return years
        avi  = [2,5,10,20,50,100,200,500,1000];

        arr = ds_ann_max.values.copy()
        # loop through a 3D array and make a new 3D array with the returns
        count,rows,cols = arr.shape
        out = np.zeros((len(avi), rows, cols ), dtype=np.float32)
        for i,j in np.ndindex(arr.shape[-2:]):
            out[:,i,j] = return_intervals( arr[:,i,j], avi ).astype(np.float32)
        
        out[np.isnan(out)] = -9999

        meta = {'dtype':'float32', 
                'count':len(avi),
                'height':rows,
                'width':cols,
                'driver':'GTiff',
                'compress':'lzw',
                'crs':rasterio.crs.CRS.from_string(wrf_crs),
                'transform':a,
                'nodata':-9999
                }

        with rasterio.open('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/return_intervals_{}_{}.tif'.format(data_group,duration), 'w', **meta ) as rst:
            rst.write(out)

        # now from here we can put those into a new NetCDF file

