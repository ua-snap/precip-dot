#######################################
# Convert a GeoTiff file to NetCDF.
######################################

import xarray as xr
import rasterio
import numpy as np
from affine import Affine
from pyproj import Proj, transform

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
    import sys

    if len(sys.argv) != 3:
        print("usage: python3 tif2nc.py IN_FILE OUT_FILE")
        exit(1)

    inputfile = sys.argv[1]
    outputfile = sys.argv[2]

    with rasterio.open(inputfile) as tif_file:
        tif_meta = tif_file.meta.copy()
        tif_shape = tif_file.shape
        tif_arr = tif_file.read(1).astype(np.float32)
    
    xc, yc = coordinates(meta=tif_meta, numpy_array=tif_arr)
    out_xc = xc[0,]
    out_yc = yc[:,0]

    out_ds = xr.Dataset(
        {
            'value' : (['yc','xc'], tif_arr)
        },
        coords= {
            'xc': out_xc,
            'yc': out_yc
        }
    )

    out_ds.to_netcdf(outputfile)

