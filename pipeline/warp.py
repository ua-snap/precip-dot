# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Warp grid of WRF data to match NOAA Atlas 14 grid
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

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
    import os, glob, itertools
    import rasterio
    import numpy as np
    from rasterio.warp import reproject, Resampling
    import xarray as xr
    # import multiprocessing as mp
    import argparse

    # parse some args
    parser = argparse.ArgumentParser( description='Compute deltas for historical vs. projected data.' )
    parser.add_argument( "-p", "--path", action='store', dest='path', type=str, help="input directory storing the return interval data." )
    parser.add_argument( "-a", "--atlas", action='store', dest='atlas_path', type=str, help="input directory storing Atlas 14 data")
    parser.add_argument( "-o", "--out_path", action='store', dest='out_path', type=str, help="output directory to write outputs" )
    parser.add_argument( "-d", "--data_group", action='store', dest='data_group', type=str, help="name of the model to use: either 'NCAR-CCSM4' or 'GFDL-CM3'")

    # parse the args and unpack
    args = parser.parse_args()
    path = args.path
    atlas_path = args.atlas_path
    out_path = args.out_path
    data_group = args.data_group

    print(" Loading and parsing initial data...", flush=True)

    # Load grid from one of the ATLAS files as a template
    template_fn = os.path.join(atlas_path,'ak2yr30da.tif')
    with rasterio.open( template_fn ) as tmp:
        atlas_meta = tmp.meta.copy()
        atlas_meta.update(compress='lzw',count=1,dtype=np.float32)
        tmp_shape = tmp.shape

    # Get WRF input files
    wrf_files = glob.glob(os.path.join(path,'*_{}_*_deltas.nc'.format(data_group)))

    # Use the first one as a template to create the grid metadata for the WRF grids.
    first_wrf = wrf_files[0]
    print(" Using {} as initial file.".format(os.path.basename(first_wrf)), flush=True )
    ds = xr.open_dataset(first_wrf)

    res = 20000 # 20km resolution
    x0,y0 = np.array( ds.xc.min()-(res/2.)), np.array(ds.yc.max()+(res/2.) ) # origin point upper-left corner from centroid
    wrf_crs = '+units=m +proj=stere +lat_ts=64.0 +lon_0=-152.0 +lat_0=90.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000'
    wrf_transform = rasterio.transform.from_origin( x0, y0, res, res ) # build affine transform using rasterio mechanics

    def warp_arr(wrf_data):
        destination = np.empty(tmp_shape, dtype=np.float32)
        reproject(
            wrf_data, destination,
            src_transform=wrf_transform, dst_transform=atlas_meta['transform'],
            src_crs=wrf_crs, dst_crs=atlas_meta['crs'],
            src_nodata=-9999, dst_nodata=None,
            resampling=Resampling.bilinear,
            num_threads=4, init_dest_nodata=True
        )
        return destination

    print(" Performing initial warp...", flush=True)

    # Do an initial warp of the first interval from the first file to get the
    # shape and coordinates of the output.
    first_data = ds['pf'][0,...,...].values
    warped = warp_arr(first_data)
    xc, yc = coordinates(meta=atlas_meta, numpy_array=warped)
    out_xc = xc[0,]
    out_yc = yc[:,0]

    ds.close()

    def warp_file(fn):
        print(" {}...".format(os.path.basename(fn)), flush=True)
        wrf_ds = xr.open_dataset(fn)
        intervals = wrf_ds.interval

        # Iterate through each variable: pf, pf-upper and pf-lower
        # and fill out new dataset
        output_data = {}
        for var in ['pf', 'pf-upper', 'pf-lower']:
            output_arr = np.empty(
                [len(intervals), len(out_yc), len(out_xc)],
                dtype=np.float32
            )

            # Iterate through each interval and warp grid
            for i in range(len(intervals)):
                arr = wrf_ds[var][i,...,...].values
                warped = warp_arr(arr)
                output_arr[i] = warped

            output_data[var] = output_arr

        # Create output dataset
        out_ds = xr.Dataset(
            {
                'pf'        : (['interval','yc','xc'], output_data['pf']),
                'pf-upper'  : (['interval','yc','xc'], output_data['pf-upper']),
                'pf-lower'  : (['interval','yc','xc'], output_data['pf-lower'])
            },
            coords= {
                'xc'        : out_xc,
                'yc'        : out_yc,
                'interval'  : intervals
            }
        )
        # Write output to NetCDF
        out_fn = os.path.join(out_path, os.path.basename(fn).replace('_deltas.nc','_warp.nc'))
        out_ds.to_netcdf(out_fn)

        wrf_ds.close()

    for file in wrf_files:
        warp_file(file)
