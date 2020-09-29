# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Fudge final values to ensure that all data at a single point
# consistently increases as duration and return interval increase.
#
# At any instance of the data where the value for a duration is lower than
# the duration below it (e.g. The pf estimate for 4 days is less than the
# estimate for 3 days), we set the lower value to be equal to 1.01 times the
# higher value, thus ensuring that the values consistently increase.
#
# We alternate doing this iterating over durations and then intervals and
# repeat the process until no more values need to be adjusted.
#
# By the way, if you're reading this and thinking "hmm, that doesn't seem
# statistically sound", you're right. But it's what NOAA did with their
# data so we get to do it too!
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import os, glob
import argparse
import xarray as xr
import numpy as np

path = None
out_path = None
data_group = None

DURATIONS = ['60m','2h', '3h', '6h', '12h','24h','3d','4d','7d','10d','20d','30d','45d','60d',]

# Keep track of which durations we've already covered. On the first time using
# each duration, load the file from the input directory and for later iterations
# load the file from the output directory.
visited_durations = [False for _ in DURATIONS]

def iterate_durations():
    print(" Iterating over durations...", flush=True)

    values_changed = 0

    # Iterate over pairs of durations (higher and lower)
    for (i, higher_dur) in enumerate(DURATIONS[1:], start=1):
        lower_dur = DURATIONS[i-1]
        print(f" {lower_dur} -> {higher_dur}", flush=True)

        if visited_durations[i]:
            src_dir = out_path
        else:
            src_dir = path
            visited_durations[i]   = True
            visited_durations[i-1] = True

        # Durations are stored in separate files, so we need to get the corresponding
        # files for the higher and lower durations
        higher_files = glob.glob(os.path.join(src_dir,f"*_{data_group}_sum_wrf_{higher_dur}_*.nc"))

        for higher_file in higher_files:
            lower_file = higher_file.replace(higher_dur,lower_dur)

            # Load data
            higher_ds = xr.open_dataset(higher_file)
            lower_ds  = xr.open_dataset(lower_file)
            higher_arr = higher_ds['pf'].values
            lower_arr  = lower_ds['pf'].values

            # mask represents the indices where the higher duration is less
            # than the lower duration
            diff = higher_arr - lower_arr
            mask = diff < 0

            # Adjust values
            if len(mask) > 0:
                values_changed += len(mask)
                higher_arr[mask] = lower_arr[mask] * 1.01
                higher_ds['pf'][...,...,...] = higher_arr

            # Save results
            higher_file_out = os.path.join(out_path,os.path.basename(higher_file))
            higher_ds.to_netcdf(higher_file_out)
            if i == 1:
                lower_file_out = os.path.join(out_path,os.path.basename(lower_file))
                lower_ds.to_netcdf(lower_file_out)

            higher_ds.close()
            lower_ds.close()

    return values_changed


if __name__ == '__main__':

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

    num = iterate_durations()
    print(f"    {num} value(s) changed.", flush=True)
