# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Multiply NOAA Atlas 14 data by deltas to get final precipitation estimates
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == '__main__':
    import os, glob
    import re
    import xarray as xr
    import rasterio
    import numpy as np
    # import multiprocessing as mp
    import argparse

    # parse some args
    parser = argparse.ArgumentParser( description='Compute deltas for historical vs. projected data.' )
    parser.add_argument( "-p", "--path", action='store', dest='path', type=str, help="input directory storing the return interval data." )
    parser.add_argument( "-a", "--atlas", action='store', dest='atlas_path', type=str, help="input directory storing NOAA Atlas 14 data")
    parser.add_argument( "-o", "--out_path", action='store', dest='out_path', type=str, help="output directory to write outputs" )
    parser.add_argument( "-d", "--data_group", action='store', dest='data_group', type=str, help="name of the model to use: either 'NCAR-CCSM4' or 'GFDL-CM3'" )
    
    # parse the args and unpack
    args = parser.parse_args()
    path = args.path
    atlas_path = args.atlas_path
    out_path = args.out_path
    data_group = args.data_group

    # names of the durations in the deltas files
    DURATIONS       = ['60m','2h', '3h', '6h', '12h','24h','3d','4d','7d','10d','20d','30d','45d','60d',]
    # Those same durations, but as they are named in the NOAA files ('2d' is missing)
    DURATIONS_NOAA  = ['01h','02h','03h','06h','12h','24h','3d','4d','7d','10d','20d','30d','45d','60d',]
    # Intervals as they are named in NOAA files
    INTERVALS       = ['2yr', '5yr', '10yr', '25yr', '50yr', '100yr', '200yr', '500yr', '1000yr']

    interval_regex = re.compile(r'^ak(\d+yr)')
    # Get the index of the interval for a NOAA Atlas file.
    def interval_index(filename):
        base = os.path.basename(filename)
        interval = interval_regex.match(base).group(1)
        return INTERVALS.index(interval)

    TIMESLICES = [ ('2020','2049'), ('2050','2079'), ('2080','2099') ]

    for (d, d_noaa) in zip(DURATIONS, DURATIONS_NOAA):
        print(" duration: {}".format(d), flush=True)

        # Get NOAA Atlas files for this duration
        atlas_files         = glob.glob(os.path.join(atlas_path,'ak*{}a_ams.tif'.format(d_noaa)))
        # Ana sort by interval
        atlas_files.sort(key=interval_index)

        for ts in ["{}-{}".format(x[0],x[1]) for x in TIMESLICES]:
            print("  time period: {}".format(ts), flush=True)

            # Find the (downscaled) deltas file. (There should only be one)
            deltas_file = glob.glob(os.path.join(path,'*_{}_*_{}_{}*_warp.nc'.format(data_group,d,ts)))[0]

            ds = xr.open_dataset(deltas_file)

            # Iterate through each return interval
            for i in range(len(ds.interval)):
                arr       = ds['pf'      ][i,...,...].values
                arr_upper = ds['pf-upper'][i,...,...].values
                arr_lower = ds['pf-lower'][i,...,...].values
                with rasterio.open(atlas_files[i]) as tmp:
                    atlas_arr = tmp.read(1).astype(np.float32)

                # Multiply data
                multiplied = arr * atlas_arr
                below_threshold = multiplied < 0
                multiplied[below_threshold] = float('nan')
                ds['pf'][i,...,...] = multiplied

            # Save file
            out_fn = os.path.join(out_path,os.path.basename(deltas_file).replace('_warp.nc','_multiply.nc'))
            ds.to_netcdf(out_fn)

            ds.close()

