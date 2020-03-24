# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Make deltas for historical vs projected data
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def run(duration):

    # Get historical data
    # (for the historical data, there should only be one file for this duration)
    hist_file = glob.glob(os.path.join(path,'*_duration_*{}_historical_*.nc'.format(duration,data_group)))[0]
    hist_ds   = xr.open_dataset(hist_file)

    # Get projected date
    # (There should be one for each decade for this duration)
    proj_files = glob.glob(os.path.join(path,'*_duration_*{}_rcp85_*.nc'.format(duration,data_group)))

    # Compute Deltas
    for fn in proj_files:
        proj_ds = xr.open_dataset(proj_ds)

        proj_ds /= hist_ds

        out_fn = fn.replace('_intervals.nc', '_deltas.nc')
        proj_ds.to_netcdf(out_fn)

        proj_ds.close()

    hist_ds.close()


if __name__ == '__main__':
    import os, glob, itertools
    import xarray as xr
    # import multiprocessing as mp
    import argparse

    # parse some args
    parser = argparse.ArgumentParser( description='Compute deltas for historical vs. projected data.' )
    parser.add_argument( "-p", "--path", action='store', dest='path', type=str, help="input directory storing the return interval data." )
    parser.add_argument( "-o", "--out_path", action='store', dest='out_path', type=str, help="output directory to write outputs" )
    parser.add_argument( "-d", "--data_group", action='store', dest='data_group', type=str, help="name of the model to use: either 'NCAR-CCSM4' or 'GFDL-CM3'" )
    
    # parse the args and unpack
    args = parser.parse_args()
    path = args.path
    out_path = args.out_path
    data_group = args.data_group

    DURATIONS_NOAA = ['60m','2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d',] # names of the durations in the NOAA tool

    for d in DURATIONS_NOAA:
        print(" duration: {}".format(d))
        run(d)