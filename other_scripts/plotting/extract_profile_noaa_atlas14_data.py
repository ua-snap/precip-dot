
if __name__ == '__main__':
    import os, argparse, affine
    import pyproj
    import pandas as pd

    parser = argparse.ArgumentParser( description='extract lat/lon point to NOAA Atlas 14-style table and return as Pandas DataFrame' )
    parser.add_argument( "-lat", "--lat", action='store', dest='lat', type=float, help="latitude WGS84" )
    parser.add_argument( "-lon", "--lon", action='store', dest='lon', type=float, help="longitude WGS84" )

    # unpack args
    args = parser.parse_args()
    lat = args.lat
    lon = args.lon

    # # # TESTING
    # lon,lat = (-147.7164,64.8378) # # FOR TESTING
    # # # TESTING

    fn = '/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/netcdf/noaa_atlas14_ak_missing2d.nc'
    template_fn = '/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/raw/warped/ak1yr01ha_3338.tif'
    with rasterio.open(template_fn) as tmp:
        a = tmp.transform
    
    p2 = pyproj.Proj(init='epsg:3338')
    x1,y1 = p2(lon, lat)

    a = meta['transform']
    colrows = [ np.array(~a*(x,y)).astype(np.int).tolist() for x,y in shp_polar.geometry.apply(lambda x: (x.x, x.y))]

    # slicing and showing the table we want to see for a single variable.
    table = new_ds.pr_freq[...,row,col].to_pandas() / 1000 # rescale data too...
