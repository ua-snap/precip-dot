# make a dataframe of ERA-Interim precip values that can be used in 
# 	GEV analysis testing across the WRF grid. 

if __name__ == '__main__':
	import pandas as pd
	import geopandas as gpd
	import numpy as np
	import os,glob,rasterio
	import xarray as xr

	path = '/rcs/project_data/wrf_data/hourly_fix/pcpt'
	os.chdir( path )

	# load the data
	group = 'NCAR-CCSM4_historical' #'ERA-Interim_historical', 'GFDL-CM3_historical'
	ds = xr.open_mfdataset('./*{}*.nc'.format(group))

	ds_sum = ds.resample(time='D').sum().compute()

	# some constant metadata about the WRF grid
	res = 20000 # 20km resolution
	# origin point upper-left corner from centroid
	x0,y0 = np.array( ds.xc.min()-(res/2.)), np.array(ds.yc.max()+(res/2.) )
	wrf_crs = '+units=m +proj=stere +lat_ts=64.0 +lon_0=-152.0 +lat_0=90.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000'
	a = rasterio.transform.from_origin( x0, y0, res, res ) # build affine transform using rasterio mechanics

	# open up the shapefile of points used in testing
	shp_fn = '/workspace/Shared/Tech_Projects/DOT/project_data/ancillary/alaska_test_locations.shp'
	shp = gpd.read_file( shp_fn )

	# reproject to WRF polar
	shp_polar = shp.to_crs(wrf_crs)
	rowcols = {shp_polar['NameX'].iloc[idx]:np.array(~a*(i[0],i[1])).astype(np.int).tolist() for idx,i in enumerate(shp_polar.geometry.apply(lambda x: (x.x, x.y)))}

	# pull the vals
	vals = {i:ds_sum.pcpt[:,rowcols[i][0],rowcols[i][1]].values for i in rowcols}
	# names = shp_polar.NameX.tolist()

	# fairbanks_rowcol = [141, 125]
	# pts_df = pd.DataFrame(dict(zip(names, vals)), index=ds.time.to_index())
	pts_df = pd.DataFrame(vals, index=ds_sum.time.to_index())
	pts_df.to_csv('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/pcpt_day_sum_communities_{}.csv'.format(group))
