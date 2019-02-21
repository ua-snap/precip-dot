# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # make annual maximum series (AMS) from the WRF Precipitation Hourlies
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def make_ams( ds, duration ):
	return ds.resample(time='1D').max()

def ams_aggregation( fn, duration, out_fn ):
	# read in the dataset
	ds = xr.open_mfdataset(fn)

	# make the AMS
	out_ds = make_ams( ds, duration )
	
	# dump to disk
	out_ds.to_netcdf( out_fn )
	return out_fn

def run( x ):
	return ams_aggregation(*x)

if __name__ == '__main__':
	import os, glob, itertools
	import xarray as xr
	import multiprocessing as mp
	
	out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt'
	path = '/workspace/Shared/Tech_Projects/wrf_data/project_data/wrf_data/hourly/pcpt'
	durations_noaa_name = ['60m','2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d',]
	durations_pandas_str = ['1H','2H','3H','6H','12H','1D','2D','3D', '4D', '7D', '10D', '20D', '30D', '45D', '60D'] # these are PANDAS date strings

	files = glob.glob( os.path.join( path, '*.nc' ) )

	# make the arguments:
	args = []
	for fn, duration in itertools.product( files, durations_pandas_str ):
		out_fn = os.path.join(out_path, os.path.basename(fn).replace('.nc', '_{}_ams.nc'.format(duration)))
		args = args + [(fn,duration,out_fn)]

	# run in parallel
	ncpus = 5
	try:
		pool = mp.Pool( ncpus )
		out = pool.map( run, args )
		pool.close()
		pool.join()
	except:
		# make sure the pool is closed
		pool.close()
		pool.join()
		pool = None
