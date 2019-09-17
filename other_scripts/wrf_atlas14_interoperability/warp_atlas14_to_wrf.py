
def reproject_to_wrf(fn, meta, out_path):
	atlas14_crs = {'init':'epsg:4269'}
	with rasterio.open(fn) as rst:
		src_crs = atlas14_crs
		src_nodata = np.nan
		src_transform = rst.transform
		source = rst.read(1).astype(np.float32)
		source[ source < 0 ] = np.nan
		source = source / 1000 # make it a float from the scaled values

	destination = np.zeros((meta['height'], meta['width']), dtype=np.float32)
	reproject(source, destination, src_transform=src_transform,
			src_crs=src_crs, src_nodata=src_nodata, dst_transform=meta['transform'], 
			dst_crs=meta['crs'], dst_nodata=np.nan,resampling=Resampling.average, 
			num_threads=4, init_dest_nodata=True,)

	out_fn = os.path.join(out_path, os.path.basename(fn).replace('.asc','.tif'))
	with rasterio.open(out_fn,'w',**meta) as out_rst:
		out_rst.write(destination, 1)
	return out_fn 

if __name__ == '__main__':
	import rasterio
	from rasterio.warp import reproject, Resampling
	import numpy as np
	import os,glob
	import multiprocessing as mp
	from functools import partial

	# list the files
	files = glob.glob('/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/raw/extracted/*.asc')
	out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/warped_to_wrf'

	template_fn = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_interval_durations/pf_ERA-Interim_historical_2yr30d.tif'
	with rasterio.open( template_fn ) as tmp:
		meta = tmp.meta.copy()
		meta.update(compress='lzw',count=1,dtype=np.float32,driver='GTiff',nodata=np.nan)
		tmp_shape = tmp.shape

	f = partial(reproject_to_wrf, meta=meta, out_path=out_path)
	pool = mp.Pool(15)
	out = pool.map(f, files)
	pool.close()
	pool.join()
