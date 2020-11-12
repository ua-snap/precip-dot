##############################################
# For a given pair of integer indices,
# print a table summarizing the annual maximum series
# data at that point.
##############################################

import sys, os
import xarray as xr

DATADIR     = "/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/ams/"

DURATIONS   = [ '60m','2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d' ]
DATASETS = [ 
    'GFDL-CM3_historical', 'NCAR-CCSM4_historical',
    'GFDL-CM3_rcp85', 'NCAR-CCSM4_rcp85'
]
TIMESLICES  = [ ('2020','2049'), ('2050','2079'), ('2080','2099') ]

if len(sys.argv) != 3:
    print("usage: point_data.py XCOORD YCOORD", file=sys.stderr)
    exit(1)

x = int(sys.argv[1])
y = int(sys.argv[2])

for dataset in DATASETS:

    times = []
    if "historical" in dataset:
        times = [('1979','2015')]
    else:
        times = TIMESLICES

    for ts in times:
        ts_str = f"{ts[0]}-{ts[1]}"

        for duration in DURATIONS:
            print(f"{dataset} {ts_str} ({duration}):\t\t", end='')

            ds = xr.open_dataset(os.path.join(
                DATADIR,
                f"pcpt_{dataset}_sum_wrf_{duration}_{ts_str}_ams.nc")
            )

            arr = ds['pcpt'].values[...,x,y]
            print(" ".join([f"{(x):.3f}".ljust(8) for x in arr]))

            ds.close()

        print("")
