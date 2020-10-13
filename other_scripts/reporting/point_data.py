##############################################
# For a given pair of xc and yc coordinates,
# print a table summarizing the final data
# at that point.
##############################################

import sys, os
import xarray as xr

DATADIR     = "/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/undiff/"

DURATIONS   = [ '60m','2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d' ]
DATASETS    = [ 'GFDL-CM3', 'NCAR-CCSM4' ]
TIMESLICES  = [ ('2020','2049'), ('2050','2079'), ('2080','2099') ]
VARIABLES   = [ 'pf-upper', 'pf', 'pf-lower']
INTERVALS   = [2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

if len(sys.argv) != 3:
    print("usage: point_data.py XCOORD YCOORD", file=sys.stderr)
    exit(1)

x = float(sys.argv[1])
y = float(sys.argv[2])

for dataset in DATASETS:
    for duration in DURATIONS:
        print(f"Duration: {duration}    Dataset: {dataset}")
        header = (
            "                        "+
            " ".join([str(x).ljust(8) for x in INTERVALS])
        )
        print(header)
        print("="*len(header))

        for ts in TIMESLICES:
            ts_str = f"{ts[0]}-{ts[1]}"
            ds = xr.open_dataset(os.path.join(
                DATADIR,
                f"pcpt_{dataset}_sum_wrf_{duration}_{ts_str}_undiff.nc")
            )

            for var in VARIABLES:
                arr = ds.sel(xc=[x],yc=[y],method="nearest")[var].values[...,0,0]
                print(
                    ts_str +
                    f" ({var})".ljust(15) +
                    " ".join([f"{(x/1000):.3f}".ljust(8) for x in arr])
                )

            ds.close()
            print("")
        print("")
