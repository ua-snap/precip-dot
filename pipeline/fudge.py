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

import os
from glob import glob
import argparse
import xarray as xr
import numpy as np

path = None
out_path = None
data_group = None

DURATIONS = ['60m','2h', '3h', '6h', '12h','24h', '2d', '3d','4d','7d','10d','20d','30d','45d','60d',]
visited_durations = [False for _ in DURATIONS]

# Iterate over all of the durations and fudge values to ensure they consistently
# increase as duration increases. Returns the number of values that had to be
# changed
def iterate_durations():
    print(" Iterating over durations...", flush=True)

    values_changed = 0

    # Iterate over pairs of durations (higher and lower)
    for (i, higher_dur) in enumerate(DURATIONS[1:], start=1):
        lower_dur = DURATIONS[i-1]
        print(f" {lower_dur} -> {higher_dur}", flush=True)

        # Naturally, we want to load files from the input directory and save
        # them into the output directory, but since each file is likely
        # to be loaded/saved multiple times as we change values, we need to be
        # sure to only load from the input directory the first time and from the
        # output directory all other times (so that changes to values persist
        # as we do multiple iterations). So we track which durations we've
        # "visited" before and choose the directory to load from accordingly.
        if visited_durations[i]:
            src_dir = out_path
        else:
            src_dir = path
            visited_durations[i]   = True
            visited_durations[i-1] = True

        higher_files = glob(os.path.join(src_dir,f"*_{data_group}_sum_wrf_{higher_dur}_*.nc"))

        for higher_file in higher_files:
            # Durations are stored across files, so we need to get the corresponding
            # files for the higher and lower durations
            lower_file = higher_file.replace(higher_dur,lower_dur)

            # Load data
            higher_ds = xr.load_dataset(higher_file)
            lower_ds  = xr.load_dataset(lower_file)
            higher_arr = higher_ds['pf'].values
            lower_arr  = lower_ds['pf'].values

            # mask represents the indices where the higher duration is less
            # than the lower duration
            diff = higher_arr - lower_arr
            mask = diff <= 0
            num = np.count_nonzero(mask)

            # Adjust values
            if num > 0:
                print(f"    {num} value(s) changed", flush=True)
                values_changed += num
                higher_arr[mask] = lower_arr[mask] * 1.01
                higher_ds['pf'][...,...,...] = higher_arr

            # Save results
            higher_file_out = os.path.join(
                out_path,os.path.basename(higher_file)\
                    .replace('_multiply.nc','_fudge.nc')
            )
            higher_ds.to_netcdf(higher_file_out)

            # The lower duration doesn't get changed, but we need to ensure that
            # the lowest duration (60m) is still in the output directory, so
            # we simply save it back out on the first pass.
            if i == 1:
                lower_file_out = os.path.join(
                    out_path,os.path.basename(lower_file)\
                    .replace('_multiply.nc','_fudge.nc')
                )
                lower_ds.to_netcdf(lower_file_out)

            higher_ds.close()
            lower_ds.close()


    return values_changed

# Iterate over all of the intervals and fudge values to ensure they consistently
# increase as interval increases. Returns the number of values that had to be
# changed
def iterate_intervals():
    print(" Iterating over intervals...", flush=True)

    values_changed = 0

    # Since intervals are stored within each file, this process is
    # more straightforward than iterating over durations since all the info
    # we need to compare interval values are contained within a single file.
    # We just have to work on each file in turn.
    # Since the first iteration over intervals occurs after the iteration over
    # durations, we can also just load from the output directory since the
    # durations step would have moved everything into there.

    files = glob(os.path.join(out_path,f"*_{data_group}_*.nc"))

    for file in files:
        print(f" {os.path.basename(file)}", flush=True)

        ds = xr.load_dataset(file)

        for i in range(1, len(ds.interval)):
            higher_arr = ds['pf'][i  ,...,...].values
            lower_arr  = ds['pf'][i-1,...,...].values

            diff = higher_arr - lower_arr
            mask = diff <= 0
            num = np.count_nonzero(mask)

            if num > 0:
                print(f"    {num} value(s) changed", flush=True)
                values_changed += num
                higher_arr[mask] = lower_arr[mask] * 1.01
                ds['pf'][i,...,...] = higher_arr

        ds.close()
        ds.to_netcdf(file)

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

    while True:
        num = 0
        num += iterate_durations()
        num += iterate_intervals()
        if num == 0:
            break

    print(" Done!", flush=True)

