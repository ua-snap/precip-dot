# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # PERFORM EXTREME VALUE ANALYSIS (EVA) USING L-MOMENTS 
# # OF AMS DATA PRECOMPUTED FOR GIVEN DURATIONS.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # -- NOTES -- :
# # Great online resources doing exactly what we want to do:
# # https://www.linkedin.com/pulse/beginners-guide-carry-out-extreme-value-analysis-2-chonghua-yin/
# # https://github.com/royalosyin/A-Beginner-Guide-to-Carry-out-Extreme-Value-Analysis-with-Codes-in-Python

def return_intervals(dat, avi):
    if not (dat == 0).all():
        # get LMOMENTS
        # LMU = lmom.lmom_ratios(dat)
        paras = distr.gev.lmom_fit(dat)
        fitted_gev = distr.gev(**paras)

        pavi = np.empty(len(avi))
        # pavi2 = np.empty(len(avi))
        for i in range(len(avi)):
            pavi[i] = 1.0-(1.0 / avi[i]) # good one
            # pavi2[i] = np.exp(-(1/avi[i])) # testing...

        # return year precip estimates
        return fitted_gev.ppf(pavi)
    else:
        return np.repeat(np.nan, len(avi))

if __name__ == '__main__':
    # import matplotlib
    # matplotlib.use('agg')
    # from matplotlib import pyplot as plt
    import rasterio, itertools, glob, os
    import xarray as xr
    import lmoments3 as lmom
    from lmoments3 import distr
    import pandas as pd
    import geopandas as gpd
    import numpy as np

    # set-up pathing and list files
    path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/singlefile/ams'
    out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_interval_durations'
    files = sorted(glob.glob(os.path.join(path,'pcpt_*_ams.nc')))
    files = [ fn for fn in files if 'rcp85' not in fn ]

    for fn in files:
        # read in precalculated annual maximum seried
        ds = xr.open_mfdataset( fn ).load()
        ds_ann_max = ds['pcpt'] / 25.4 # make inches...
        
        # some constant metadata about the WRF grid
        res = 20000 # 20km resolution
        x0,y0 = np.array( ds.xc.min()-(res/2.)), np.array(ds.yc.max()+(res/2.) ) # origin point upper-left corner from centroid
        wrf_crs = '+units=m +proj=stere +lat_ts=64.0 +lon_0=-152.0 +lat_0=90.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000'
        a = rasterio.transform.from_origin( x0, y0, res, res ) # build affine transform using rasterio mechanics
        
        # set some return years
        # avi = [1,2,5,10,20,50,100,200,500,1000]
        avi = [1,2,5,10,25,50,100,200,500,1000]

        arr = ds_ann_max.values.copy()
        # loop through a 3D array and make a new 3D array with the returns
        count,rows,cols = arr.shape
        out = np.zeros((len(avi), rows, cols ), dtype=np.float32)
        for i,j in np.ndindex(arr.shape[-2:]):
            out[:,i,j] = return_intervals( arr[:,i,j], avi ).astype(np.float32)
        
        out[np.isnan(out)] = -9999

        # dump it to a rasterio raster GeoTiff file (multiband) in order of return years above.
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
        
        out_fn = os.path.join(out_path, 'multiband', os.path.basename(fn).replace('_ams.nc', '_multiband_intervals.tif'))
        with rasterio.open(out_fn , 'w', **meta ) as rst:
            rst.write(out)

        # now from here we can put those into new output singleband GTiffs
        intervals = ['0'+str(i) if len(str(i)) == 1 else str(i) for i in avi ]
        for i in np.arange(out.shape[0]):
            meta.update(count=1)
            variable, model, scenario, metric, project, duration, junk = os.path.basename(fn).split('.')[0].split('_')
            out_base_fn = 'pf_{}_{}_{}yr{}.tif'.format(model,scenario,intervals[i],duration)
            out_fn = os.path.join(out_path, out_base_fn)
            with rasterio.open( out_fn, 'w', **meta ) as rst:
                rst.write( out[i,...], 1 )


