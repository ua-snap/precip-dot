# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # make cascading durations series which are used in AMS computation for 
# # DOT project using the WRF Precipitation Hourlies.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# some global naming and duration args
DURATIONS_NOAA = ['2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d',] # names of the durations in the NOAA tool
DURATIONS_PANDAS = ['2H','3H','6H','12H','1D','2D','3D','4D','7D','10D','20D','30D','45D','60D',] # these are PANDAS date strings
OUT_NAMING_LU = dict(zip(DURATIONS_PANDAS, DURATIONS_NOAA))
# '60m','1H', <- removed from durations since they are handled differently

def run_duration( files, duration, out_fn, variable='pcpt' ):
    '''
    run the duration resampling and dump to disk

    args:
        files = [list/str] a list of sorted chronological file paths, a single file path, 
                        or a glob pattern in a string path to read in NetCDF files using MFDataset
        duration = [str] pandas datetime grouper string
        out_fn = [str] path to the new output NetCDF file to be serialized to disk.
    
    returns:
        path to the new NetCDF file created.

    '''
    ds = xr.open_mfdataset(files).load()
    ds_dur = ds.resample(time=duration).sum()
    # out_fn = os.path.join(out_path, '{0}_{1}_sum_wrf_{2}.nc'.format(variable,OUT_NAMING_LU[duration], group))

    # compression encoding
    encoding = ds_dur[ variable ].encoding
    encoding.update( zlib=True, complevel=5, contiguous=False, chunksizes=None, dtype='float32' )
    ds_dur[ variable ].encoding = encoding

    # write to disk
    ds_dur.to_netcdf(out_fn)
    ds_dur.close(); ds_dur=None
    ds.close(); ds=None
    return out_fn

def run_short_duration( fn, duration, out_fn, variable='pcpt' ):
    ''' 
    A method to run the shorter length durations on longer series where RAM is an issue.
    Dask appears to still choke on memory when dealing with lots of hourlies on disk and 
    trying to resample to say 2 or 3 hourly. This method just operates on each year and 
    dumps them to disk instead of concatenating to a larger NetCDF. 
    '''
    
    # load the file / resample to duration
    ds = xr.open_dataset(fn).load()
    ds_dur = ds.resample(time=duration).sum()
    
    # compression encoding
    encoding = ds_dur[ variable ].encoding
    encoding.update( zlib=True, complevel=5, contiguous=False, chunksizes=None, dtype='float32' )
    ds_dur[ variable ].encoding = encoding

    # write to disk
    ds_dur.to_netcdf(out_fn)
    ds_dur.close(); ds_dur=None
    ds.close(); ds=None
    return out_fn

if __name__ == '__main__':
    import os, glob, dask, shutil
    import xarray as xr
    import pandas as pd
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
    variable = 'pcpt'

    # # # # # TESTING:
    # path = '/rcs/project_data/wrf_data/hourly_fix/pcpt'
    # out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/durations'
    # data_group = 'NCAR-CCSM4_rcp85'
    # # # # # END TESTING

    # sort/group the filenames, toss in a dict
    files = sorted(glob.glob(os.path.join(path, '*{}*.nc'.format(data_group) )))
    if 'rcp85' in data_group:
        # filter to the rcp85 < 2060...
        years = [ int(os.path.basename(fn).split('.')[0].split('_')[-1]) for fn in files ]
        files = pd.Series(files, index=years).loc[:2059].tolist()
        wrf_files = files.copy() # keep a copy for moving 1 hourly at the end.

    # run the durations in a cascading fashion
    for duration in DURATIONS_PANDAS:
        print(' duration:{}'.format(duration))
        out_names = []
        if (duration in ['2H','3H','6H']):
            for fn in files:            
                year = fn.split('.nc')[0].split('_')[-1] # from standard naming convention
                out_fn = os.path.join(out_path, '{0}_{1}_sum_wrf_{2}_{3}.nc'.format(variable, OUT_NAMING_LU[duration], data_group, year))
                _ = run_short_duration( fn, duration, out_fn, variable='pcpt' )
                out_names = out_names + [out_fn]
        else:
            out_fn = os.path.join(out_path, '{0}_{1}_sum_wrf_{2}.nc'.format(variable,OUT_NAMING_LU[duration], data_group))
            _ = run_duration( files, duration, out_fn, variable='pcpt' )
            out_names = out_names + [out_fn]
        
        # reset the files name for the next round of durations in the cascade
        files = out_names

        # move hourly data to the output location -- it is the starting 'duration'
        print(' moving the base hourlies, renamed to final naming convention, to the output_path')
        years = [ os.path.basename(fn).split('.')[0].split('_')[-1] for fn in wrf_files ]
        out_filenames = [os.path.join(out_path, 'pcpt_{0}_sum_wrf_{1}_{2}.nc'.format('60m', data_group, year)) for year in years]
        _ = [ shutil.copy( fn, out_fn ) for fn,out_fn in zip(files, out_filenames) if not os.path.exists(out_fn) ]
