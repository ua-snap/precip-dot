# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # COMPUTE L-MOMENTS STATISTICS FOR A GIVEN DURATION/AMS 
# # AND WRITE TO multi and single BAND GeoTIFFs
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def return_lmoments( ams ):
    '''
    return the lmoments in form of: l1, l2, t3, t4, t5
    based on the values of the annual maximum series at a
    duration length.
    
    arguments:
    ---------
    ams = array of AMS values for a given location

    returns:
    -------
    lmoments in the form of: l1, l2, t3, t4, t5
    where elems 3-5 are lmoment ratios
    
    '''
    l1, l2, t3, t4, t5 = lmom.lmom_ratios(ams)
    return [l1, l2, t3, t4, t5]

def run_par_update_arr( idx ):
    '''run parallel updating of the shared c-types array '''
    i,j = idx
    tmp_out = np.ctypeslib.as_array(out_shared)
    try:
        tmp_out[:,i,j] = return_lmoments( arr[:,i,j] )#.astype(np.float32)
    except:
        # handle missing data
        tmp_out[:,i,j] = np.nan


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
    from collections import OrderedDict
    from numpy.random import randint as _randint
    import multiprocessing as mp
    from multiprocessing import sharedctypes
    import itertools

    # set-up pathing and list files
    path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/annual_maximum_series'
    out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_lmoment_grids'
    files = sorted(glob.glob(os.path.join(path,'pcpt_*_ams.nc')))
    files = [ fn for fn in files if 'rcp85' not in fn ]
    data_groups = ['ERA-Interim_historical']#,'GFDL-CM3_historical','NCAR-CCSM4_historical'] #,'GFDL-CM3_rcp85','NCAR-CCSM4_rcp85']
    durations = ['60m', '2h', '3h', '6h', '12h', '24h', '2d', '3d', '4d', '7d', '10d', '20d', '30d', '45d', '60d']
    ncpus = 63

    for group in data_groups:
        print(group)
        out_data = []
        for duration in durations:
            print(duration)
            # read in precalculated annual maximum series
            fn = os.path.join(path,'pcpt_{0}_sum_wrf_{1}_ams.nc'.format(group,duration))
            ds = xr.open_mfdataset( fn ).load()
            xc, yc = ds.xc, ds.yc
            ds_ann_max = ds['pcpt']*0.0393701 # make inches...
            lmom_list = ['l1','l2','t3','t4','t5']

            # some constant metadata about the WRF grid
            res = 20000 # 20km resolution
            x0,y0 = np.array( ds.xc.min()-(res/2.)), np.array(ds.yc.max()+(res/2.) ) # origin point upper-left corner from centroid
            wrf_crs = '+units=m +proj=stere +lat_ts=64.0 +lon_0=-152.0 +lat_0=90.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000'
            a = rasterio.transform.from_origin( x0, y0, res, res ) # build affine transform using rasterio mechanics
            
            arr = ds_ann_max.values.copy() 
            good_index = np.apply_along_axis(lambda x: (x == 0).all(),arr=arr, axis=0)
            idx = np.where(good_index == True)
            bad_idx = list(zip(*idx))

            # run in parallel
            indexes = list(np.ndindex(arr.shape[-2:]))

            # make the output arrays?
            rows,cols = arr.shape[1:]
            out = np.ctypeslib.as_ctypes(np.zeros((len(lmom_list),rows,cols), dtype=np.float))

            out_shared = sharedctypes.RawArray(out._type_, out)
     
            p = mp.Pool(ncpus)
            p.map( run_par_update_arr, indexes )
            p.close()
            p.join()

            # bring these c-types arrays back to numpy arrays.
            out = np.ctypeslib.as_array(out_shared).astype(np.float32)

            # update the np.nans to something more useable and SNAP-ish
            out[np.isnan(out)] = -9999
            out_data = out_data + [out]

        new_arr = np.array(out_data)

        # build NetCDF dataset 4D (durations, lmoments1-5, rows, cols)
        new_ds = xr.Dataset({'pr_freq': (['durations','lmom','yc', 'xc'], new_arr.astype(np.float32).copy())},
                    coords={'xc': xc,
                            'yc': yc,
                            'durations':durations,
                            'lmom':lmom_list })

        # write it back out to disk with compression encoding
        variable = 'pr_freq'
        encoding = new_ds[ variable ].encoding
        encoding.update( zlib=True, complevel=5, contiguous=False, chunksizes=None, dtype='float32' )
        new_ds[ variable ].encoding = encoding
        # make a new fn naming convention.
        
        out_fn = os.path.join(out_path, os.path.basename(fn).replace('_ams.nc', '_ams_lmoments_multiband.nc'))
        new_ds.to_netcdf(out_fn, format='NETCDF4')

        # # [example] slice out fairbanks
        # df = new_ds.pr_freq[:,:,126,140].to_pandas()

        # dump it to a rasterio raster GeoTiff file (multiband) in order of return years above.
        meta = {'dtype':'float32', 
                'count':len(lmom_list),
                'height':rows,
                'width':cols,
                'driver':'GTiff',
                'compress':'lzw',
                'crs':rasterio.crs.CRS.from_string(wrf_crs),
                'transform':a,
                'nodata':-9999
                }
        
        out_fn = os.path.join(out_path, os.path.basename(fn).replace('_ams.nc', '_ams_lmoments_multiband.tif'))
        with rasterio.open(out_fn , 'w', **meta ) as rst:
            rst.write(out)

        # now from here we can put those into new output singleband GTiffs
        # # intervals = ['0'+str(i).split('.')[0] if len(str(i)) == 1 else str(i) for i in durations ]
        for lm,i in zip(lmom_list, np.arange(out.shape[0])):
            meta.update(count=1)
            out_fn = os.path.join(out_path, os.path.basename(fn).replace('_ams.nc', '_ams_{}.tif'.format(lm)))
            with rasterio.open( out_fn, 'w', **meta ) as rst:
                rst.write( out[i,...], 1 )
