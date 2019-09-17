# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # make cascading durations series which are used in AMS computation for 
# # DOT project using the WRF Precipitation Hourlies.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# some global naming and duration args
DURATIONS_NOAA = ['2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d',] # names of the durations in the NOAA tool
DURATIONS_PANDAS = ['2H','3H','6H','12H','1D','2D','3D','4D','7D','10D','20D','30D','45D','60D',] # these are PANDAS date strings
OUT_NAMING_LU = dict(zip(DURATIONS_PANDAS, DURATIONS_NOAA))
# '60m','1H', <- removed from durations since they are handled differently

def resample_to_duration(ds, duration):
	''' 
	aggregate data along time dimension using the duration string in PANDAS format
	
	arguments:
		ds = [xarray.Dataset] dataset object to use in the summation resampling
		duration = [str] pandas datetime grouper string

	returns:
		a new xarray.Dataset object resampled to the new duration using a sum (total).

	'''
	return ds.resample(time=duration).sum().compute()

def run_group_cascading( files, out_path ):
	''' 
	run the duration resampling in a cascading way 
	
	args:
		files = [list/str] a list of sorted chronological file paths, a single file path, 
						or a glob pattern in a string path to read in NetCDF files using MFDataset
		out_path = [str] path to the output directory we want to toss the newly created files into.
	
	returns:
		a list of output filenames for each of the cascading duration lengths with the side-effect of 
		generating new NetCDF files for the durations at those listed locations.

	'''
	# filename games
	cur_fn = files 

	# loop through the durations in an successively larger order, cascading.
	out_names = []
	for duration in DURATIONS_PANDAS:
		ds_dur = run_duration( cur_fn, duration )
		out_fn = os.path.join(out_path, 'pcpt_{0}_sum_wrf_{1}.nc'.format(OUT_NAMING_LU[duration], group))
		ds_dur.to_netcdf(out_fn)
		out_names = out_names + [out_fn]
		cur_fn = out_fn
	return out_names

def run_duration( files, duration ):
	'''
	run the duration resampling in a functional way

	args:
		files = [list/str] a list of sorted chronological file paths, a single file path, 
						or a glob pattern in a string path to read in NetCDF files using MFDataset
		duration = [str] pandas datetime grouper string
	
	returns:
		new xr.Dataset object resampled to the duration length

	'''
	ds = xr.open_mfdataset(files)
	return resample_to_duration(ds, duration)


if __name__ == '__main__':
	import os, glob, dask, shutil
	import xarray as xr
	import pandas as pd
	
	# set up some pathing
	path = '/rcs/project_data/wrf_data/hourly_fix/pcpt'
	out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/durations'
	
	# set up the durations
	data_groups = ['GFDL-CM3_rcp85']#'ERA-Interim_historical',]#'GFDL-CM3_rcp85','GFDL-CM3_historical','NCAR-CCSM4_historical','NCAR-CCSM4_rcp85']

	# sort/group the filenames, toss in a dict
	file_groups = {}
	for data_group in data_groups:
		files = sorted(glob.glob(os.path.join(path, '*{}*.nc'.format(data_group) )))
		if 'rcp85' in data_group:
			# filter to the rcp85 < 2050...
			years = [ int(os.path.basename(fn).split('.')[0].split('_')[-1]) for fn in files ]
			files = pd.Series(files, index=years).loc[:2060].tolist()
		file_groups[data_group] = files

	# run the durations in a cascading fashion
	done = {}
	for group in file_groups:
		print( 'runnning: {}'.format(group))
		files = file_groups[group]
		# run it
		out_files = run_group_cascading( files, out_path )
		done[group] = out_files

		# we also need to move that hourly data to the output location
		print('moving the base hourlies, renamed to final naming convention, to the output_path')
		years = [ os.path.basename(fn).split('.')[0].split('_')[-1] for fn in files ]
		out_filenames = [os.path.join(out_path, 'pcpt_{0}_sum_wrf_{1}_{2}.nc'.format('60m', group, year)) for year in years]
		_ = [ shutil.copy( fn, out_fn ) for fn,out_fn in zip(files, out_filenames) if not os.path.exists(out_fn) ]
