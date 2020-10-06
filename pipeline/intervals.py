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

    import itertools, glob, os
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
    parser.add_argument( "-p", "--path", action='store', dest='path', type=str, help="input directory storing the Annual Maximum Series files" )
    parser.add_argument( "-o", "--out_path", action='store', dest='out_path', type=str, help="output directory to write outputs" )
    parser.add_argument( "-d", "--data_group", action='store', dest='data_group', type=str, help="name of the model/scenario we are running against. format:'NCAR-CCSM4_historical'" )
    parser.add_argument( "-n", "--ncpus", action='store', dest='ncpus', type=int, help="number of cpus to use" )
    
    # parse the args and unpack
    args = parser.parse_args()
    path = args.path
    out_path = args.out_path
    data_group = args.data_group
    ncpus = args.ncpus

    files = sorted(glob.glob(os.path.join(path,'*{}*.nc'.format(data_group) )))

    for fn in files:
        print(f" {os.path.basename(fn)}", flush=True)

        # read in precalculated annual maximum series
        ds = xr.open_mfdataset( fn, combine='by_coords' ).load()
        ds_ann_max = ds['pcpt']*0.0393701 # make inches...
        # ds_ann_max = ds['pcpt'] / 25.4 # make inches too...
        
        # set some return years
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
        
        # Create output dataset
        out_ds = xr.Dataset(
            {
                'pf'        : (['interval','xc','yc'], out),
                'pf-upper'  : (['interval','xc','yc'], out_ci_upper),
                'pf-lower'  : (['interval','xc','yc'], out_ci_lower)
            },
            coords= {
                'xc'        : ds.xc,
                'yc'        : ds.yc,
                'interval'  : avi
            }
        )

        # Write data to NetCDF
        out_fn = os.path.join(out_path, os.path.basename(fn).replace('_ams.nc','_intervals.nc'))
        out_ds.to_netcdf(out_fn)

        out_ds.close()
        ds.close()
