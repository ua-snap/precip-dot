# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Basically the inverse of interval_diffs.py
# Rewrite confidence intervals as their absolute values as opposed to their
# differences from the mean
# (oh and convert them from millimeters to thousandts of an inch)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import os, glob
import xarray as xr
import argparse

def run(fn):

    ds = xr.open_dataset(fn)

    ds['pf-upper'] = ds['pf'] + ( ds['pf-upper'] * 39.3701 )
    ds['pf-lower'] = ds['pf'] + ( ds['pf-lower'] * 39.3701 )

    out_fn = os.path.join(
        out_path,
        os.path.basename(fn)
    )
    ds.to_netcdf(out_fn)

    ds.close()

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

    files = sorted( glob.glob(os.path.join(path, f'*{data_group}*.nc')) )

    for fn in files:
        print(f" {fn}", flush=True)
        run(fn)