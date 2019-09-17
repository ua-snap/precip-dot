
def reproject_to_atlas14(fn, meta, out_path):
	with rasterio.open(fn) as rst:
		src_crs = rst.crs
		src_nodata = rst.nodata
		src_transform = rst.transform
		source = rst.read(1)

	destination = np.empty(tmp_shape, dtype=np.float32)
	reproject(source, destination, src_transform=src_transform,
			src_crs=src_crs, src_nodata=src_nodata, dst_transform=meta['transform'], 
			dst_crs=meta['crs'], dst_nodata=None,resampling=Resampling.bilinear, 
			num_threads=4, init_dest_nodata=True,)

	out_fn = os.path.join(out_path, os.path.basename(fn))
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
	files = glob.glob('/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_interval_durations/*.tif')
	out_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_interval_durations_warped'

	template_fn = '/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/raw/warped/ak2yr30da.tif'
	with rasterio.open( template_fn ) as tmp:
		meta = tmp.meta.copy()
		meta.update(compress='lzw',count=1,dtype=np.float32)
		tmp_shape = tmp.shape

	# fn = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_interval_durations/pf_ERA-Interim_historical_25yr20d.tif'
	# test = reproject_to_atlas14(fn, meta)
	f = partial(reproject_to_atlas14, meta=meta, out_path=out_path)
	pool = mp.Pool(15)
	out = pool.map(f, files)
	pool.close()
	pool.join()
