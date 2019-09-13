# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # make cascading durations series which are used in AMS computation for 
# # DOT project using the WRF Precipitation Hourlies.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# some global naming and duration args
DURATIONS_NOAA = ['2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d',] # names of the durations in the NOAA tool
DURATIONS_PANDAS = ['2H','3H','6H','12H','1D','2D','3D','4D','7D','10D','20D','30D','45D','60D',] # these are PANDAS date strings
OUT_NAMING_LU = dict(zip(DURATIONS_PANDAS, DURATIONS_NOAA))
# '60m','1H', <- removed from durations since they are handled differently

# def resample_to_duration(ds, duration):
# 	''' 
# 	aggregate data along time dimension using the duration string in PANDAS format
	
# 	arguments:
# 		ds = [xarray.Dataset] dataset object to use in the summation resampling
# 		duration = [str] pandas datetime grouper string

# 	returns:
# 		a new xarray.Dataset object resampled to the new duration using a sum (total).

# 	'''
# 	return ds.resample(time=duration).sum().compute()

# def run_group_cascading( files, out_path ):
# 	''' 
# 	run the duration resampling in a cascading way 
	
# 	args:
# 		files = [list/str] a list of sorted chronological file paths, a single file path, 
# 						or a glob pattern in a string path to read in NetCDF files using MFDataset
# 		out_path = [str] path to the output directory we want to toss the newly created files into.
	
# 	returns:
# 		a list of output filenames for each of the cascading duration lengths with the side-effect of 
# 		generating new NetCDF files for the durations at those listed locations.

# 	'''
# 	# hardwired variable
# 	variable = 'pcpt'
	
# 	# filename games :)
# 	cur_fn = files 

# 	# loop through the durations in an successively larger order, cascading.
# 	out_names = []
# 	for duration in DURATIONS_PANDAS:
# 		ds_dur = run_duration( cur_fn, duration )
# 		out_fn = os.path.join(out_path, '{0}_{1}_sum_wrf_{2}.nc'.format(variable,OUT_NAMING_LU[duration], group))

# 		# write it back out to disk with compression encoding
# 		encoding = ds_dur[ variable ].encoding
# 		encoding.update( zlib=True, complevel=5, contiguous=False, chunksizes=None, dtype='float32' )
# 		ds_dur[ variable ].encoding = encoding
		
# 		ds_dur.to_netcdf(out_fn)
# 		ds_dur.close(); ds_dur=None
# 		out_names = out_names + [out_fn]
# 		cur_fn = out_fn
# 	return out_names

def run_duration( files, duration, out_fn, variable='pcpt' ):
	'''
	run the duration resampling and dump to disk

	args:
		files = [list/str] a list of sorted chronological file paths, a single file path, 
						or a glob pattern in a string path to read in NetCDF files using MFDataset
		duration = [str] pandas datetime grouper string
		out_fn = [str] path to the new output NetCDF file to be serialized to disk.
	
	returns:
		path to the new NetCDF file created.

	'''
	ds = xr.open_mfdataset(files).load()
	ds_dur = ds.resample(time=duration).sum()
	# out_fn = os.path.join(out_path, '{0}_{1}_sum_wrf_{2}.nc'.format(variable,OUT_NAMING_LU[duration], group))

	# compression encoding
	encoding = ds_dur[ variable ].encoding
	encoding.update( zlib=True, complevel=5, contiguous=False, chunksizes=None, dtype='float32' )
	ds_dur[ variable ].encoding = encoding

	# write to disk
	ds_dur.to_netcdf(out_fn)
	ds_dur.close(); ds_dur=None
	ds.close(); ds=None
	return out_fn

def run_short_duration( fn, duration, out_fn, variable='pcpt' ):
	''' 
	A method to run the shorter length durations on longer series where RAM is an issue.
	Dask appears to still choke on memory when dealing with lots of hourlies on disk and 
	trying to resample to say 2 or 3 hourly. This method just operates on each year and 
	dumps them to disk instead of concatenating to a larger NetCDF. 
	'''
	
	# load the file / resample to duration
	ds = xr.open_dataset(fn).load()
	ds_dur = ds.resample(time=duration).sum()
	
	# compression encoding
	encoding = ds_dur[ variable ].encoding
	encoding.update( zlib=True, complevel=5, contiguous=False, chunksizes=None, dtype='float32' )
	ds_dur[ variable ].encoding = encoding

	# write to disk
	ds_dur.to_netcdf(out_fn)
	ds_dur.close(); ds_dur=None
	ds.close(); ds=None
	return out_fn

if __name__ == '__main__':
	import os, glob, dask, shutil
	import xarray as xr
	import pandas as pd
	
	# set up some pathing
	path = '/rcs/project_data/wrf_data/hourly_fix/pcpt'
	# path = '/workspace/Shared/Tech_Projects/DOT/project_data/data_s3'
	# path = '/workspace/Shared/Tech_Projects/wrf_data/project_data/wrf_data/hourly_fix/pcpt'
	# path = '/storage01/malindgren/wrf_ccsm4/hourly_fix/pcpt'
	# path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/pcpt'
	out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/durations'
	variable = 'pcpt'

	# set up the durations
	# data_groups = ['ERA-Interim_historical']
	# data_groups = ['GFDL-CM3_historical']
	# data_groups = ['NCAR-CCSM4_historical']
	# data_groups = ['GFDL-CM3_rcp85']
	data_groups = ['NCAR-CCSM4_rcp85']

	# sort/group the filenames, toss in a dict
	file_groups = {}
	for data_group in data_groups:
		files = sorted(glob.glob(os.path.join(path, '*{}*.nc'.format(data_group) )))
		if 'rcp85' in data_group:
			# filter to the rcp85 < 2050...
			years = [ int(os.path.basename(fn).split('.')[0].split('_')[-1]) for fn in files ]
			files = pd.Series(files, index=years).loc[:2059].tolist()
		file_groups[data_group] = files

	# run the durations in a cascading fashion
	for group in file_groups:
		print( 'runnning: {}'.format(group))
		files = file_groups[group]

		for duration in DURATIONS_PANDAS:
			print('running:{}'.format(duration))
			out_names = []
			if (duration in ['2H','3H','6H']):
				for fn in files:			
					year = fn.split('.nc')[0].split('_')[-1] # from standard naming convention
					out_fn = os.path.join(out_path, '{0}_{1}_sum_wrf_{2}_{3}.nc'.format(variable, OUT_NAMING_LU[duration], group, year))
					_ = run_short_duration( fn, duration, out_fn, variable='pcpt' )
					out_names = out_names + [out_fn]
			else:
				out_fn = os.path.join(out_path, '{0}_{1}_sum_wrf_{2}.nc'.format(variable,OUT_NAMING_LU[duration], group))
				_ = run_duration( files, duration, out_fn, variable='pcpt' )
				out_names = out_names + [out_fn]
			
			# reset the files name for the next round of durations in the cascade
			files = out_names

		# move hourly data to the output location -- it is the starting 'duration'
		print('moving the base hourlies, renamed to final naming convention, to the output_path')
		files = file_groups[group]
		years = [ os.path.basename(fn).split('.')[0].split('_')[-1] for fn in files ]
		out_filenames = [os.path.join(out_path, 'pcpt_{0}_sum_wrf_{1}_{2}.nc'.format('60m', group, year)) for year in years]
		_ = [ shutil.copy( fn, out_fn ) for fn,out_fn in zip(files, out_filenames) if not os.path.exists(out_fn) ]
