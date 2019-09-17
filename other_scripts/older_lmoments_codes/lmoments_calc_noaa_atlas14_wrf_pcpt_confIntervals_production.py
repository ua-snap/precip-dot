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
        sample_fitted = distr.gev(*samplefit)

        # set up the return intervals to pull from fitted distribution
        # intervals = np.arange(0.1, 1000.0, 0.1) + 1 
        # intervals = T
        sample_intervals = sample_fitted.ppf(1.0-1./intervals)
        res = samplefit
        res.extend(sample_intervals.tolist())
        return tuple(res)

    # the calculations itself
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

def plot_it(fitted_gev, dat_intervals):
    # sT = dat_ret_intervals
    N    = np.r_[1:len(dat)+1]*1.0 #must *1.0 to convert int to float
    Nmax = max(N)

    # plot it out:
    fig, ax = plt.subplots()

    plt.setp(ax.lines, linewidth = 2, color = 'magenta')

    ax.set_title("GEV Distribution")
    ax.set_xlabel("Return Period (Year)")
    ax.set_ylabel("Precipitation")
    ax.semilogx(avi, dat_intervals)
    ax.scatter(Nmax/N, sorted(dat)[::-1], color = 'orangered')

    ax.semilogx(avi, ci_Td, '--')
    ax.semilogx(avi, ci_Tu, '--')
    ax.fill_between(avi, ci_Td, ci_Tu, color = '0.75', alpha = 0.5)

    plt.savefig('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/test_confInterval.png')
    plt.close()
    plt.cla()
    return 1

def run_par_update_arr( idx ):
    tmp_out = np.ctypeslib.as_array(out_shared)
    tmp_out_ci_upper = np.ctypeslib.as_array(out_ci_upper_shared)
    tmp_out_ci_lower =np.ctypeslib.as_array(out_ci_lower_shared)

    i,j = idx
    tmp_out[:,i,j] = run( arr[:,i,j], avi )#.astype(np.float32)
    tmp = run_ci( arr[:,i,j], avi ) # clunky, but will have to do...
    tmp_out_ci_upper[:,i,j] = tmp['upper_ci']#.astype(np.float32) 
    tmp_out_ci_lower[:,i,j] = tmp['lower_ci']#.astype(np.float32)

def coordinates( fn=None, meta=None, numpy_array=None, input_crs=None, to_latlong=False ):
    '''
    take a raster file as input and return the centroid coords for each 
    of the grid cells as a pair of numpy 2d arrays (longitude, latitude)

    User must give either:
        fn = path to the rasterio readable raster
    OR
        meta & numpy ndarray (usually obtained by rasterio.open(fn).read( 1 )) 
        where:
        meta = a rasterio style metadata dictionary ( rasterio.open(fn).meta )
        numpy_array = 2d numpy array representing a raster described by the meta

    input_crs = rasterio style proj4 dict, example: { 'init':'epsg:3338' }
    to_latlong = boolean.  If True all coordinates will be returned as EPSG:4326
                         If False all coordinates will be returned in input_crs
    returns:
        meshgrid of longitudes and latitudes

    borrowed from here: https://gis.stackexchange.com/a/129857
    ''' 
    
    import rasterio
    import numpy as np
    from affine import Affine
    from pyproj import Proj, transform

    if fn:
        # Read raster
        with rasterio.open( fn ) as r:
            T0 = r.transform  # upper-left pixel corner affine transform
            p1 = Proj( r.crs )
            A = r.read( 1 )  # pixel values

    elif (metadata is not None) & (numpy_array is not None):
        A = numpy_array
        if input_crs != None:
            p1 = Proj( input_crs )
            T0 = meta[ 'transform' ]
        else:
            p1 = None
            T0 = meta[ 'transform' ]
    else:
        BaseException( 'check inputs' )

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[1]), np.arange(A.shape[0]))
    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation( 0.5, 0.5 )
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: ( c, r ) * T1
    # All eastings and northings -- this is much faster than np.apply_along_axis
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)

    if to_latlong == False:
        return eastings, northings
    elif (to_latlong == True) & (input_crs != None):
        # Project all longitudes, latitudes
        longs, lats = transform(p1, p1.to_latlong(), eastings, northings)
        return longs, lats
    else:
        BaseException( 'cant reproject to latlong without an input_crs' )


def to_nc( arr, xc, yc, intervals, durations ):
    # build dataset with levels at each timestep
    new_ds = xr.Dataset({'pr_freq': (['durations','intervals','yc', 'xc'], arr.copy())},
                        coords={'xc': ('xc', xc[0,]),
                                'yc': ('yc', yc[:,0]),
                                'intervals':intervals,
                                'durations':durations })

    # # this encoding stuff is still non-working, but would be great to finally figure out.
    # encoding = {'pr_freq':{'dtype': 'int16', 'zlib': True,'complevel': 9, 'scale_factor':.001,}} 

    # write it out to disk
    out_fn = '/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/netcdf/pr_freq_noaa-atlas14_ak_{}.nc'.format(group_name)
    new_ds.to_netcdf( out_fn, engine='scipy' )

    # cleanup
    new_ds.close()


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt

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

    # set-up pathing and list files
    path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/annual_maximum_series'
    out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_interval_durations'
    files = sorted(glob.glob(os.path.join(path,'pcpt_*_ams.nc')))
    files = [ fn for fn in files if 'rcp85' not in fn ]
    ncpus = 63

    for fn in files:
        # read in precalculated annual maximum seried
        ds = xr.open_mfdataset( fn ).load()
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
        avi = np.array([2,5,10,25,50,100,200,500,1000]).astype(np.float64)

        arr = ds_ann_max.values.copy() 
        good_index = np.apply_along_axis(lambda x: (x == 0).all(),arr=arr, axis=0)
        idx = np.where(good_index == True)
        bad_idx = list(zip(*idx))

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

        # # make NetCDF files from the outputs
        # xc,yc = coordinates(meta=meta, numpy_array=out[0,0,...])

        # to_nc( arr, xc, yc, avi, durations )
        
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

