##############################################
# For a list of final result files (that is, the 
# the output of the "combined" step.), print a CSV containing
# a series of percentiles for the values. (One row for
# each dataset/duration/decade/return interval)
##############################################

import sys, os
import xarray as xr
import numpy as np
from natsort import natsorted

if len(sys.argv) < 2:
    print("usage: percentiles.py FILE...", file=sys.stderr)
    exit(1)

filenames = sys.argv[1:]
filenames = natsorted(filenames)

PERCENTILES = [1,5,25,50,75,95,99]

print(','.join(["dataset","duration","decade","interval"]+[str(x) for x in PERCENTILES]))

for file in filenames:
    components = os.path.basename(file).split('_')
    dataset, duration, decade = [components[i] for i in [1,4,5]]

    ds = xr.open_dataset(file)
    intervals = ds.interval

    for i in range(len(intervals)):
        arr = ds['pf'][i,...,...].values
        percents = np.nanpercentile(arr,PERCENTILES).tolist()
        percents_as_str = [f"{x:.3f}" for x in percents]

        print(','.join(
            [dataset,duration,decade,str(float(intervals[i]))] + percents_as_str)
        )

    ds.close()
