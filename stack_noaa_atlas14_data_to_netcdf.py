# build 4-d NetCDF files from the downloaded NOAA Atlas14 data...

def open_raster( fn, band=1 ):
    with rasterio.open( fn ) as rst:
        arr = rst.read(1)
    return arr

def rescale_data( arr ):
    arr = arr.astype(np.float32)
    arr[arr != -9] = arr[arr != -9]/1000
    arr[arr == -9] = -9999
    return arr

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

    elif (meta is not None) & (numpy_array is not None):
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


if __name__ == '__main__':
    import os, glob, rasterio, itertools
    import xarray as xr
    import numpy as np
    import pandas as pd

    # setup
    file_path = '/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/raw/warped'
    groups = ['partialduration-upperlimit','partialduration-lowerlimit',
                'annualmaximum-upperlimit','annualmaximum-lowerlimit',
                'annualmaximum','partialduration']
    
    durations = ['01h','02h','03h','06h','12h','24h','03d','04d','07d','10d','20d','30d','45d','60d',] # '02d', <- appear to me missing from the series...

    for group_name in groups:
        out = []
        print(group_name)
        for duration in durations:
            files = glob.glob(os.path.join( file_path, '*yr{}*.tif'.format(duration)) )
            cur_files = []
            for fn in files:
                sub_fn = os.path.basename(fn).split('.')[0]

                # filters for different data groups
                filter_files = {'partialduration-upperlimit':sub_fn.endswith('u'),
                                 'partialduration-lowerlimit':sub_fn.endswith('l'),
                                 'annualmaximum-upperlimit':sub_fn.endswith('u_ams'),
                                 'annualmaximum-lowerlimit':sub_fn.endswith('l_ams'),
                                 'annualmaximum':not sub_fn.endswith('u_ams') and 
                                                    not sub_fn.endswith('l_ams') and 
                                                    '_ams' in sub_fn ,
                                 'partialduration':not sub_fn.endswith('u') and 
                                                    not sub_fn.endswith('l') and 
                                                    '_ams' not in sub_fn }
                if filter_files[group_name]:
                    cur_files = cur_files + [fn]
            
            df = pd.DataFrame([ os.path.basename(fn).split('.')[0].split('yr') for fn in cur_files ], columns=['interval', 'duration'])
            if 'annualmaximum' in group_name: # no annual for 1year interval (obviously)
                interval_order_ak = ['ak2','ak5','ak10','ak25','ak50','ak100','ak200','ak500','ak1000',]
                intervals = [2,5,10,25,50,100,200,500,1000] # nyears
            else:
                interval_order_ak = ['ak1','ak2','ak5','ak10','ak25','ak50','ak100','ak200','ak500','ak1000',]
                intervals = [1,2,5,10,25,50,100,200,500,1000] # nyears
            
            # order the files
            ordered_files = [list(filter( lambda x: interval+'yr' in x, cur_files )) for interval in interval_order_ak]
            cur_files = [j for i in ordered_files for j in i ] # unpack
                
            # stack the current duration and store in a list
            arr = np.array([ open_raster(fn) for fn in cur_files ])
            out = out + [arr.copy()]
            del arr

        # stack these new 3d arrays to 4d
        arr = np.array(out).astype(np.int32)

        # get coordinates:
        with rasterio.open(cur_files[0]) as tmp:
            meta = tmp.meta.copy()

        xc,yc = coordinates(meta=meta, numpy_array=arr[0,0,...])

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
        del new_ds
        del arr
