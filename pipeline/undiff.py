# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Basically the inverse of interval_diffs.py
# Rewrite confidence intervals as their absolute values as opposed to their
# differences from the mean
# (oh and convert them from inches to thousandts of an inch)
#
# Since this is the last step in the pipeline we also prepare the data 
# to be the final product by adding header information and renaming variables
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import os, glob
import xarray as xr
import argparse
import re
from datetime import datetime

timerange_regex = re.compile(r'^.*?(20\d\d\-20\d\d)_\w+\.nc$')
def get_timerange(fn):
    '''
    Extract and return the timerange (e.g. 2020-2049) from the given filename.
    '''
    return timerange_regex.match(fn).group(1)

datagroup_regex = re.compile(r'pcpt_(.+?)_sum_wrf')
def get_datagroup(fn):
    '''
    Extract and return the datagroup (e.g GFDL-CM3) from the given filename
    '''
    return datagroup_regex.match(fn).group(1)

def run(fn):

    ds = xr.open_dataset(fn)

    # Create dataset
    ds_out = xr.Dataset(
        {
            'pf'        : ( ('interval','yc','xc'), ds['pf']                         ),
            'pf_upper'  : ( ('interval','yc','xc'), ds['pf'] + (ds['pf-upper']*1000) ),
            'pf_lower'  : ( ('interval','yc','xc'), ds['pf'] + (ds['pf-lower']*1000) )
        },
        coords = {
            'xc'        : ds.xc,
            'yc'        : ds.yc,
            'interval'  : ds.interval
        }
    )

    # Variable descriptions
    ds_out['pf']      .attrs['long_name'] = "AMS-based precipitation frequency estimates"
    ds_out['pf_upper'].attrs['long_name'] = "Upper 95% confidence bounds on AMS-based precipitation frequency estimates"
    ds_out['pf_lower'].attrs['long_name'] = "Lower 95% confidence bounds on AMS-based precipitation frequency estimates"
    # Variable units
    ds_out['pf']      .attrs['units'] = "1/1000 inches"
    ds_out['pf_upper'].attrs['units'] = "1/1000 inches"
    ds_out['pf_lower'].attrs['units'] = "1/1000 inches"

    # Coordinate descriptions
    ds_out.interval.attrs['long_name']     = "Annual exceedance probability"
    ds_out.interval.attrs['units']         = "1/years"
    ds_out.xc      .attrs['long_name']     = "X-coordinate in projected coordinate system"
    ds_out.xc      .attrs['standard_name'] = "projection_x_coordinate"
    ds_out.yc      .attrs['long_name']     = "Y-coordinate in projected coordinate system"
    ds_out.yc      .attrs['standard_name'] = "projection_y_coordinate"

    # Global Attributes
    ds_out.attrs["Conventions"]      = "CF"
    ds_out.attrs["institution"]      = "Scenarios Network for Alaska + Arctic Planning"
    ds_out.attrs["contact"]          = "kmredilla@alaska.edu"
    ds_out.attrs["history"]          =  "{} Python".format(str(datetime.utcnow()))
    ds_out.attrs["comment"]          = "Intended for use by Alaska Department of Transportation"

    datagroup = get_datagroup(os.path.basename(fn))
    timerange = get_timerange(os.path.basename(fn))
    ds_out.attrs["source_gcm"]       = datagroup
    ds_out.attrs["source_timerange"] = timerange

    # CRS
    ds_out['crs'] = int()
    ds_out['crs'].attrs['grid_mapping_name'] = "albers_conical_equal_area"
                                                # LONG STRING WARNING!!
    ds_out['crs'].attrs['crs_wkt']           = 'PROJCS["NAD83 / Alaska Albers",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["standard_parallel_1",55],PARAMETER["standard_parallel_2",65],PARAMETER["latitude_of_center",50],PARAMETER["longitude_of_center",-154],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3338"]]'
    ds_out['pf']      .attrs['grid_mapping'] = 'crs'
    ds_out['pf_upper'].attrs['grid_mapping'] = 'crs'
    ds_out['pf_lower'].attrs['grid_mapping'] = 'crs'

    out_fn = os.path.join(
        out_path,
        os.path.basename(fn)\
            .replace('_fudge.nc','_undiff.nc')
    )
    ds_out.to_netcdf(out_fn)

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
        print(f" {os.path.basename(fn)}", flush=True)
        run(fn)