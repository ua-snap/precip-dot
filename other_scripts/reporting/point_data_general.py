##############################################
# For a given pair of integer indices,
# and the specification of a data directory,
# print a table summarizing the data for that
# step at that point.
#
# In practice, this will only work for the 
# directories intervals, diff, deltas, warp, multiply
# fudge and undiff
##############################################

import sys, os
import xarray as xr

DATADIR     = "/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/"

DURATIONS   = [ '60m','2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d' ]
DATASETS    = [ 'GFDL-CM3', 'NCAR-CCSM4' ]
TIMESLICES  = [ ('2020','2049'), ('2050','2079'), ('2080','2099') ]
VARIABLES   = [ 'pf-upper', 'pf', 'pf-lower']
INTERVALS   = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

if len(sys.argv) != 4:
    print("usage: point_data.py DIR XCOORD YCOORD", file=sys.stderr)
    exit(1)

step = sys.argv[1]
x = int(sys.argv[2])
y = int(sys.argv[3])

DATADIR = os.path.join(DATADIR,step)
if not os.path.isdir(DATADIR):
    print("directory not found: ", DATADIR)
    exit(1)

# We need the split-up datasets if we're looking at the return intervals
use_return_intervals = (
    step.startswith('intervals') or step.startswith('diff')
)
if use_return_intervals:
    DATASETS = [ 
        'GFDL-CM3_historical', 'NCAR-CCSM4_historical',
        'GFDL-CM3_rcp85', 'NCAR-CCSM4_rcp85'
    ]

# Undiff renames the variables too
use_undiff = step.startswith('undiff')
if use_undiff:
    VARIABLES = [ 'pf_upper', 'pf', 'pf_lower']

FILE_SUFFIX = step.replace('/','').split('-')[0]


for dataset in DATASETS:

    times = []
    if "historical" in dataset:
        times = [('1979','2015')]
    else:
        times = TIMESLICES

    for duration in DURATIONS:
        print(f"Duration: {duration}    Dataset: {dataset}")
        header = (
            "                        "+
            " ".join([str(x).ljust(8) for x in INTERVALS])
        )
        print(header)
        print("="*len(header))

        for ts in times:
            ts_str = f"{ts[0]}-{ts[1]}"
            ds = xr.open_dataset(os.path.join(
                DATADIR,
                f"pcpt_{dataset}_sum_wrf_{duration}_{ts_str}_{FILE_SUFFIX}.nc")
            )

            for var in VARIABLES:
                arr = ds[var].values[...,y,x]
                print(
                    ts_str +
                    f" ({var})".ljust(15) +
                    " ".join([f"{(x):.3f}".ljust(8) for x in arr])
                )

            ds.close()
            print("")
        print("")
