##############################################
# Get the average width of the confidence intervals (pf-upper minus pf-lower)
# For the given return intervals file
##############################################

import sys, os
import xarray as xr
import numpy as np

if len(sys.argv) != 2:
    print("usage: get_average_ci_width.py FILE", file=sys.stderr)
    exit(1)

filename = sys.argv[1]

ds = xr.open_dataset(filename)
diff = ds['pf-upper'] - ds['pf-lower']

avg = np.average(diff.values)

print(avg)
