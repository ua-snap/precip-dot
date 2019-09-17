# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # make annual maximum series (AMS) from the WRF-derived PCPT durations
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def make_ams( ds ):
	return ds.resample(time='Y').max().compute()

def run( x ):
	fn, out_fn = x
	ds = xr.open_mfdataset( fn )
	ds_ams = make_ams( ds )
	ds_ams.to_netcdf( out_fn )
	ds.close()
	ds_ams.close()
	ds = None
	ds_ams = None
	return out_fn

if __name__ == '__main__':
	import os, glob, itertools, dask
	import xarray as xr
	# import multiprocessing as mp
		
	# pathing
	path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/singlefile'
	out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/singlefile/ams'

	DURATIONS_NOAA = ['60m','2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d',] # names of the durations in the NOAA tool
	data_groups = ['GFDL-CM3_rcp85'] #['GFDL-CM3_historical','ERA-Interim_historical',] # ['GFDL-CM3_rcp85']
	# build args
	file_groups = [sorted(glob.glob(os.path.join(path,'*_{}_*{}*.nc').format(duration, group))) for duration, group in itertools.product(DURATIONS_NOAA, data_groups)]
	# out_files = [files[0].replace('singlefile','singlefile/ams').replace('.nc', '_ams.nc') for files in file_groups]
	out_files = [os.path.join(out_path,'pcpt_{}_sum_wrf_{}_ams.nc'.format(group, duration)) for duration, group in itertools.product(DURATIONS_NOAA, data_groups)]

	args = list(zip(file_groups,out_files))

	out = [] 
	for x in args:
		print(x[0])
		out = out + [run(x)]

	# # run in parallel
	# ncpus = 2
	# try:
	# 	pool = mp.Pool( ncpus )
	# 	out = pool.map( run, args )
	# 	pool.close()
	# 	pool.join()
	# except:
	# 	# make sure the pool is closed
	# 	pool.close()
	# 	pool.join()
	# 	pool = None