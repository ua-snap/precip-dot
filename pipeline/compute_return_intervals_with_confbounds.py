# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # PERFORM EXTREME VALUE ANALYSIS (EVA) USING L-MOMENTS 
# # OF AMS DATA PRECOMPUTED FOR GIVEN DURATIONS.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # -- NOTES -- :
# # Great online resources doing exactly what we want to do:
# # https://www.linkedin.com/pulse/beginners-guide-carry-out-extreme-value-analysis-2-chonghua-yin/
# # https://github.com/royalosyin/A-Beginner-Guide-to-Carry-out-Extreme-Value-Analysis-with-Codes-in-Python

NSAMPLES=5000
def run_bootstrap(dat, lmom_fitted, intervals, method='pi'):
    ''' 
    Calculate confidence intervals using parametric bootstrap and the
    percentil interval method
    This is used to obtain confidence intervals for the estimators and
    the return values for several return values.    
    More info about bootstrapping can be found on:
        - https://github.com/cgevans/scikits-bootstrap
        - Efron: "An Introduction to the Bootstrap", Chapman & Hall (1993)
        - https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

    parametric bootstrap for return levels and parameters   
    '''
    
    # function to bootstrap
    def sample_return_intervals(data, intervals=intervals):
        # get a random sampling of the fitted distribution
        sample = lmom_fitted.rvs(len(dat))
        # return the fitted params of this new randomized sample        
        paras = distr.gev.lmom_fit(sample)
        # paras = dist_types[distribution].lmom_fit(sample) # watch this it can get you into trouble
        
        samplefit = [ paras[i] for i in ['loc', 'scale', 'c']]
        # sample_fitted = dist_types[distribution](**paras)
        sample_fitted = distr.gev(**paras)

        # set up the return intervals to pull from fitted distribution
        # intervals = np.arange(0.1, 1000.0, 0.1) + 1 
        # intervals = T
        sample_intervals = sample_fitted.ppf(1.0-1./intervals)
        res = samplefit
        res.extend(sample_intervals.tolist())
        return tuple(res)

    # the calculation
    out = boot.ci(dat, statfunction=sample_return_intervals, \
                        alpha=0.05, n_samples=NSAMPLES, \
                        method=method, output='lowhigh')
    
    ci_Td = out[0, 3:]
    ci_Tu = out[1, 3:]
    params_ci = OrderedDict()
    params_ci['c']    = (out[0,0], out[1,0])
    params_ci['loc'] = (out[0,1], out[1,1])
    params_ci['scale']    = (out[0,2], out[1,3])
    
    return {'ci_Td':ci_Td, 'ci_Tu':ci_Tu, 'params_ci':params_ci}


# def ci_bootstrap(dat, fitted_gev, intervals):
#     # Calculate confidence intervals using parametric bootstrap and the
#     # percentil interval method
#     # This is used to obtain confidence intervals for the estimators and
#     # the return values for several return values.    
#     # More info about bootstrapping can be found on:
#     #     - https://github.com/cgevans/scikits-bootstrap
#     #     - Efron: "An Introduction to the Bootstrap", Chapman & Hall (1993)
#     #     - https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

#     # parametric bootstrap for return levels and parameters   

#     # The function to bootstrap     
#     def func(data):
#         sample = fitted_gev.rvs(len(dat))
#         # sample = lmoments.randgev(len(df.index), gevfit)
        
#         paras = distr.gev.lmom_fit(sample)
#         samgevfit = [ paras[i] for i in ['loc', 'scale', 'c']]
#         samfitted_gev = distr.gev(**paras)
#         # samgevfit = lmoments.pelgev(lmoments.samlmu(sample))      
#         # 

#         # T = np.arange(0.1, 999.1, 0.1) + 1
#         # intervals = np.array([1,2,5,10,25,50,100,200,500,1000])
#         sT = samfitted_gev.ppf(1.0-1./intervals)
#         res = samgevfit
#         res.extend(sT.tolist())
#         return tuple(res)

#     # the calculations itself
#     out = bootstrap_ci(dat, statfunction = func, n_samples = 500)
#     ci_Td = out[0, 3:]
#     ci_Tu = out[1, 3:]
#     params_ci = OrderedDict()
#     params_ci['c'] = (out[0,0], out[1,0])
#     params_ci['loc'] = (out[0,1], out[1,1])
#     params_ci['scale'] = (out[0,2], out[1,3])
    
#     return {'ci_Td':ci_Td, 'ci_Tu':ci_Tu, 'params_ci':params_ci}

def fit_values(dat):
    paras = distr.gev.lmom_fit(dat)
    return distr.gev(**paras)

def return_intervals(fitted_gev, avi):
    try:
        return fitted_gev.ppf( 1.0-(1.0 / avi))
    except:
        return np.repeat(np.nan, len(avi))

def run( dat, intervals ):
    try:
        fitted_gev = fit_values(dat)
        dat_intervals = return_intervals(fitted_gev, intervals)
        return dat_intervals
    except ValueError: # handle mostly the halo pixels around the outside of the WRF data.
        return np.repeat(np.nan, len(intervals))

def run_ci( dat, intervals ):
    try:
        fitted_gev = fit_values(dat)
        # bootout = ci_bootstrap(dat, fitted_gev, intervals)
        bootout = run_bootstrap(dat, fitted_gev, intervals)
        ci_Td     = bootout["ci_Td"]
        ci_Tu     = bootout["ci_Tu"]
        params_ci = bootout["params_ci"]

        return {'lower_ci':ci_Td, 'upper_ci':ci_Tu, 
                'params_ci':params_ci}
    except ValueError:
        nan_arr = np.repeat(np.nan, len(intervals))
        return {'lower_ci':nan_arr, 'upper_ci':nan_arr, 
                'params_ci':None}

def run_par_update_arr( idx ):
    tmp_out = np.ctypeslib.as_array(out_shared)
    tmp_out_ci_upper = np.ctypeslib.as_array(out_ci_upper_shared)
    tmp_out_ci_lower =np.ctypeslib.as_array(out_ci_lower_shared)

    i,j = idx
    if (i,j) not in bad_idx:
        tmp_out[:,i,j] = run( arr[:,i,j], avi )#.astype(np.float32)
        tmp = run_ci( arr[:,i,j], avi ) # clunky, but will have to do...
        tmp_out_ci_upper[:,i,j] = tmp['upper_ci']#.astype(np.float32)
        tmp_out_ci_lower[:,i,j] = tmp['lower_ci']#.astype(np.float32)

if __name__ == '__main__':

    import rasterio, itertools, glob, os
    import xarray as xr
    import lmoments3 as lmom
    from lmoments3 import distr
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import warnings as _warnings
    from collections import OrderedDict
    from numpy.random import randint as _randint
    import multiprocessing as mp
    from multiprocessing import sharedctypes
    from collections import OrderedDict
    import scikits.bootstrap as boot
    import argparse

    # parse some args
    parser = argparse.ArgumentParser( description='stack the hourly outputs from raw WRF outputs to NetCDF files of hourlies broken up by year.' )
    parser.add_argument( "-fn", "--fn", action='store', dest='fn', type=str, help="path to the Annual Maximum Series file being run" )
    parser.add_argument( "-o", "--out_path", action='store', dest='out_path', type=str, help="output directory to write outputs" )
    parser.add_argument( "-n", "--ncpus", action='store', dest='ncpus', type=int, help="number of cpus to use" )
    
    # parse the args and unpack
    args = parser.parse_args()
    fn = args.fn
    out_path = args.out_path
    ncpus = args.ncpus


    # # set-up pathing and list files
    # path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/annual_maximum_series'
    # out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_interval_durations'
    # files = sorted(glob.glob(os.path.join(path,'pcpt_*_ams.nc')))
    # # files = [ fn for fn in files if 'rcp85' not in fn ]
    # ncpus = 63


    # read in precalculated annual maximum series
    ds = xr.open_mfdataset( fn, combine='by_coords' ).load()
    ds_ann_max = ds['pcpt']*0.0393701 # make inches...
    # ds_ann_max = ds['pcpt'] / 25.4 # make inches too...
            
    # some constant metadata about the WRF grid
    res = 20000 # 20km resolution
    x0,y0 = np.array( ds.xc.min()-(res/2.)), np.array(ds.yc.max()+(res/2.) ) # origin point upper-left corner from centroid
    wrf_crs = '+units=m +proj=stere +lat_ts=64.0 +lon_0=-152.0 +lat_0=90.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000'
    a = rasterio.transform.from_origin( x0, y0, res, res ) # build affine transform using rasterio mechanics
    
    # set some return years
    # avi = [1,2,5,10,20,50,100,200,500,1000]
    # avi = np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,5,10,25,50,100,200,500,1000]).astype(np.float64)
    avi = np.array([2,5,10,25,50,100,200,500,1000]).astype(np.float)

    # Filter out the coordinates on the grid where the values are all zeroes.
    # (The return interval calculation chokes on a distribution of all 
    # zeroes).
    # These tend to just be the coordinates along the edges of the grid,
    # so it's a normal part of the data.
    arr = ds_ann_max.values.copy() 
    good_index = np.apply_along_axis(lambda x: (x == 0).all(),arr=arr, axis=0)
    idx = np.where(good_index == True)
    bad_idx = set(zip(*idx))

    # # SERIAL APPROACH -- ~14h per file run time.:
    # # loop through a 3D array and make a new 3D array with the returns
    # count,rows,cols = arr.shape
    # out = np.ctypeslib.as_ctypes(np.zeros((len(avi), rows, cols )))
    # out_ci_upper = np.zeros((len(avi), rows, cols ), dtype=np.float32)
    # out_ci_lower = np.zeros((len(avi), rows, cols ), dtype=np.float32)
    
    # for i,j in np.ndindex(arr.shape[-2:]):
    #     if (i,j) not in bad_idx:
    #         out[:,i,j] = run( arr[:,i,j], avi ).astype(np.float32)
    #         tmp = run_ci( arr[:,i,j], avi ) # clunky, but will have to do...
    #         out_ci_upper[:,i,j] = tmp['upper_ci'].astype(np.float32)
    #         out_ci_lower[:,i,j] = tmp['lower_ci'].astype(np.float32)

    # # # # 
    # # PARALLEL APPROACH: 
    indexes = list(np.ndindex(arr.shape[-2:]))

    # make the output arrays?
    count,rows,cols = arr.shape
    out = np.ctypeslib.as_ctypes(np.zeros((len(avi),rows,cols),dtype=np.float))
    out_ci_upper = np.ctypeslib.as_ctypes(np.zeros((len(avi),rows,cols),dtype=np.float))
    out_ci_lower = np.ctypeslib.as_ctypes(np.zeros((len(avi),rows,cols),dtype=np.float))

    out_shared = sharedctypes.RawArray(out._type_, out)
    out_ci_upper_shared = sharedctypes.RawArray(out_ci_upper._type_, out_ci_upper)
    out_ci_lower_shared = sharedctypes.RawArray(out_ci_lower._type_, out_ci_lower)

    p = mp.Pool(ncpus)
    p.map( run_par_update_arr, indexes )
    p.close()
    p.join()

    # bring these c-types arrays back to numpy arrays.
    out = np.ctypeslib.as_array(out_shared).astype(np.float32)
    out_ci_upper = np.ctypeslib.as_array(out_ci_upper_shared).astype(np.float32)
    out_ci_lower = np.ctypeslib.as_array(out_ci_lower_shared).astype(np.float32)

    # update the np.nans to something more useable and SNAP-ish
    out[np.isnan(out)] = -9999
    out_ci_upper[np.isnan(out_ci_upper)] = -9999
    out_ci_lower[np.isnan(out_ci_lower)] = -9999
    
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

    mb_dirname = os.path.join(out_path,'multiband')
    if not os.path.exists(mb_dirname):
        _ = os.makedirs(mb_dirname)
    
    out_fn = os.path.join(out_path, 'multiband', os.path.basename(fn).replace('_ams.nc', '_multiband_intervals.tif'))
    with rasterio.open(out_fn , 'w', **meta ) as rst:
        rst.write(out)

    out_fn = os.path.join(out_path, 'multiband', os.path.basename(fn).replace('_ams.nc', '_multiband_intervals_upper-ci.tif'))
    with rasterio.open(out_fn , 'w', **meta ) as rst:
        rst.write(out_ci_upper)

    out_fn = os.path.join(out_path, 'multiband', os.path.basename(fn).replace('_ams.nc', '_multiband_intervals_lower-ci.tif'))
    with rasterio.open(out_fn , 'w', **meta ) as rst:
        rst.write(out_ci_lower)

    # now from here we can put those into new output singleband GTiffs
    # intervals = ['0'+str(i).split('.')[0] if len(str(i)) == 1 else str(i) for i in avi ]
    intervals = avi.astype(int).astype(str)
    for i in np.arange(out.shape[0]):
        meta.update(count=1)
        variable, model, scenario, metric, project, duration, junk = os.path.basename(fn).split('.')[0].split('_')
        out_base_fn = 'pf_{}_{}_{}yr{}.tif'.format(model,scenario,intervals[i],duration)
        out_fn = os.path.join(out_path, out_base_fn)
        with rasterio.open( out_fn, 'w', **meta ) as rst:
            rst.write( out[i,...], 1 )

        # upper-ci
        out_base_fn = 'pf_upper-ci_{}_{}_{}yr{}.tif'.format(model,scenario,intervals[i],duration)
        out_fn = os.path.join(out_path, out_base_fn)
        with rasterio.open( out_fn, 'w', **meta ) as rst:
            rst.write( out_ci_upper[i,...], 1 )

        # lower-ci
        out_base_fn = 'pf_lower-ci_{}_{}_{}yr{}.tif'.format(model,scenario,intervals[i],duration)
        out_fn = os.path.join(out_path, out_base_fn)
        with rasterio.open( out_fn, 'w', **meta ) as rst:
            rst.write( out_ci_lower[i,...], 1 )

