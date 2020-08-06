##############################################
# For a given pair of x and y coordinates (epsg:3338),
# Print a summary of the NOAA Atlas 14 Data for that
# point
##############################################

import sys, os
import re
import glob
import rasterio

DATADIR     = "/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/NOAA/"

DURATIONS   = [ '01h','02h','03h','06h','12h','24h','3d','4d','7d','10d','20d','30d','45d','60d' ]
INTERVALS   = [ '2yr', '5yr', '10yr', '25yr', '50yr', '100yr', '200yr', '500yr', '1000yr' ]

if len(sys.argv) != 3:
    print("usage: point_data.py XCOORD YCOORD", file=sys.stderr)
    exit(1)

x = float(sys.argv[1])
y = float(sys.argv[2])

interval_regex = re.compile(r'^ak(\d+yr)')
# Get the index of the interval for a NOAA Atlas file.
def interval_index(filename):
    base = os.path.basename(filename)
    interval = interval_regex.match(base).group(1)
    return INTERVALS.index(interval)

# Get the value at the x and y coordinates in the given
# ATLAS file
def get_value_at_coord(filename):
    with rasterio.open(filename) as tmp:
        data = tmp.read(1)
        row, col = tmp.index(x, y)
        return data[row, col]

for dataset in [ 'AMS', 'PD' ]:
    if dataset == 'AMS':
        file_suffix = '_ams.tif'
    else:
        file_suffix = '.tif'
    print(f"Data Source: {dataset}")

    header = (
        "            "+
        " ".join([x.ljust(8) for x in INTERVALS])
    )
    print(header)
    print("="*len(header))

    for duration in DURATIONS:
        # Get NOAA Atlas files for this duration
        atlas_files         = glob.glob(os.path.join(DATADIR,f'ak*{duration}a_ams.tif'))
        atlas_files_lower   = glob.glob(os.path.join(DATADIR,f'ak*{duration}al_ams.tif'))
        atlas_files_upper   = glob.glob(os.path.join(DATADIR,f'ak*{duration}au_ams.tif'))
        # And sort by interval
        atlas_files.sort(key=interval_index)
        atlas_files_lower.sort(key=interval_index)
        atlas_files_upper.sort(key=interval_index)

        atlas_vals          = map(get_value_at_coord, atlas_files      )
        atlas_vals_lower    = map(get_value_at_coord, atlas_files_lower)
        atlas_vals_upper    = map(get_value_at_coord, atlas_files_upper)

        print(duration+" (upper) "+" ".join([f"{(x/1000):.3f}".ljust(8) for x in atlas_vals_upper]))
        print(duration+"         "+" ".join([f"{(x/1000):.3f}".ljust(8) for x in atlas_vals      ]))
        print(duration+" (lower) "+" ".join([f"{(x/1000):.3f}".ljust(8) for x in atlas_vals_lower]))
        print("")
