# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Warp grid of WRF data to match NOAA Atlas 14 grid
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == '__main__':
    import os, glob, itertools
    import rasterio
    import numpy as np
    from rasterio import reproject, Resampling
    import xarray as xr
    # import multiprocessing as mp
    import argparse

    # parse some args
    parser = argparse.ArgumentParser( description='Compute deltas for historical vs. projected data.' )
    parser.add_argument( "-p", "--path", action='store', dest='path', type=str, help="input directory storing the return interval data." )
    parser.add_argument( "-a", "--atlas", action='store', dest='atlas_path', type=str, help="input directory storing Atlas 14 data")
    parser.add_argument( "-o", "--out_path", action='store', dest='out_path', type=str, help="output directory to write outputs" )
    parser.add_argument( "-d", "--data_group", action='store', dest='data_group', type=str, help="name of the model to use: either 'NCAR-CCSM4' or 'GFDL-CM3'" )

    # parse the args and unpack
    args = parser.parse_args()
    path = args.path
    atlas_path = args.atlas_path
    out_path = args.out_path
    data_group = args.data_group

    # Load grid from one of the ATLAS files as a template
    template_fn = os.path.join(atlas_path,'ak2yr30da.tif')
    with rasterio.open( template_fn ) as tmp:
        atlas_meta = tmp.meta.copy()
        atlas_meta.update(compress='lzw',count=1,dtype=np.float32)
        tmp_shape = tmp.shape

    # Get WRF input files
    wrf_files = glob.glob(os.path.join(path,'*_{}_rcp85_*_deltas.nc'.format(data_group)))

    # Use the first one as a template to create the grid metadata for the WRF grids.
    first_wrf = wrf_files[0]
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

    def warp_file(fn):
        wrf_ds = xr.open_dataset(fn)




