# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # make annual maximum series (AMS) from the WRF-derived PCPT durations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def make_ams( ds ):
    return ds.resample(time='Y').max().compute()

def run(duration):

    # build args
    file_names = sorted(glob.glob(os.path.join(path,'*_{}_*{}*.nc').format(duration, data_group)))
    #out_files = [os.path.join(out_path,'pcpt_{}_sum_wrf_{}_ams.nc'.format(data_group, duration)) for duration in DURATIONS_NOAA]

    ds = xr.open_mfdataset(file_names, combine='by_coords' )
    ds_ams = make_ams( ds )
    
    # write it back out to disk with compression encoding
    encoding = ds_ams[ 'pcpt' ].encoding
    encoding.update( zlib=True, complevel=5, contiguous=False, chunksizes=None, dtype='float32' )
    ds_ams[ 'pcpt' ].encoding = encoding

    timeslices = [('1979','2015')]
    # For rcp85 data (2006-2100), we split up results by decade
    if ('rcp85' in data_group):
        timeslices = [(str(x),str(x+9)) for x in range(2020,2100,10)]

    for ts in timeslices:
        ds_slice = ds_ams.sel(time=slice(ts[0],ts[1]))
        out_fn = os.path.join(out_path,'pcpt_{}_sum_wrf_{}_{}-{}_ams.nc'.format(data_group, duration, ts[0], ts[1]))
        ds_slice.to_netcdf(out_fn)

    ds.close(); ds = None
    ds_ams.close(); ds_ams = None

if __name__ == '__main__':
    import os, glob, itertools
    import xarray as xr
    # import multiprocessing as mp
    import argparse

    # parse some args
    parser = argparse.ArgumentParser( description='stack the hourly outputs from raw WRF outputs to NetCDF files of hourlies broken up by year.' )
    parser.add_argument( "-p", "--path", action='store', dest='path', type=str, help="input directory storing the chronological NetCDF files for a given variable" )
    parser.add_argument( "-o", "--out_path", action='store', dest='out_path', type=str, help="output directory to write outputs" )
    parser.add_argument( "-d", "--data_group", action='store', dest='data_group', type=str, help="name of the model/scenario we are running against. format:'NCAR-CCSM4_historical'" )
    # parser.add_argument( "-n", "--ncpus", action='store', dest='ncpus', type=int, help="number of cpus to use" )
    
    # parse the args and unpack
    args = parser.parse_args()
    path = args.path
    out_path = args.out_path
    data_group = args.data_group
    # ncpus = args.ncpus

    DURATIONS_NOAA = ['60m','2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d',] # names of the durations in the NOAA tool

    for d in DURATIONS_NOAA:
        print(" duration: {}".format(d))
        run(d)
