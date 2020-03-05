# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # make annual maximum series (AMS) from the WRF-derived PCPT durations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def make_ams( ds ):
    return ds.resample(time='Y').max().compute()

def run( x ):
    fn, out_fn = x
    ds = xr.open_mfdataset( fn, combine='by_coords' )
    ds_ams = make_ams( ds )
    
    # write it back out to disk with compression encoding
    encoding = ds_ams[ 'pcpt' ].encoding
    encoding.update( zlib=True, complevel=5, contiguous=False, chunksizes=None, dtype='float32' )
    ds_ams[ 'pcpt' ].encoding = encoding
    
    ds_ams.to_netcdf( out_fn )
    ds.close(); ds = None
    ds_ams.close(); ds_ams = None
    return out_fn

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

    # # # # # testing
    # path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/durations'
    # out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/annual_maximum_series'
    # data_group = 'NCAR-CCSM4_historical'
    # # ncpus = 32
    # # # # end testing

    DURATIONS_NOAA = ['60m','2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d',] # names of the durations in the NOAA tool
    
    # build args
    file_groups = [sorted(glob.glob(os.path.join(path,'*_{}_*{}*.nc').format(duration, data_group))) for duration in DURATIONS_NOAA]
    out_files = [os.path.join(out_path,'pcpt_{}_sum_wrf_{}_ams.nc'.format(data_group, duration)) for duration in DURATIONS_NOAA]
    args = list(zip(file_groups,out_files))

    # run in serial
    out = [] 
    for x in args:
        out = out + [run(x)]

    # # run in parallel [non-working due to some RAM process overheads]
    # ncpus = 2
    # try:
    #   pool = mp.Pool( ncpus )
    #   out = pool.map( run, args )
    #   pool.close()
    #   pool.join()
    # except:
    #   # make sure the pool is closed
    #   pool.close()
    #   pool.join()
    #   pool = None
