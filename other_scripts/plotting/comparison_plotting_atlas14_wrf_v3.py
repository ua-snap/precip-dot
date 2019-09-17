# comparison plotting
import os, glob
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
import geopandas as gpd

# some setup
base_path = '/workspace/Shared/Tech_Projects/DOT/project_data'
durations = ['60m','2h','3h','6h','12h','24h','3d','4d','7d','10d','20d','30d','45d','60d',] #'2d'
intervals = [2,5,10,25,50,100,200,500,1000] # 1
# this is for getting some meta info
ds = xr.open_dataset(os.path.join(base_path,'NOAA_Atlas14','netcdf','pr_freq_noaa-atlas14_ak_annualmaximum_wrf_grid.nc'))

# points 
# some constant metadata about the WRF grid
res = 20000 # 20km resolution
# origin point upper-left corner from centroid
x0,y0 = np.array( ds.xc.min()-(res/2.)), np.array(ds.yc.max()+(res/2.) )
wrf_crs = '+units=m +proj=stere +lat_ts=64.0 +lon_0=-152.0 +lat_0=90.0 +x_0=0 +y_0=0 +a=6370000 +b=6370000'
a = rasterio.transform.from_origin( x0, y0, res, res ) # build affine transform using rasterio mechanics
ds.close(); del ds

# open up the shapefile of points used in testing
shp_fn = '/workspace/Shared/Tech_Projects/DOT/project_data/ancillary/alaska_test_locations.shp'
shp = gpd.read_file( shp_fn )

# reproject to WRF polar
shp_polar = shp.to_crs(wrf_crs)
rowcols = {shp_polar['NameX'].iloc[idx]:np.array(~a*(i[0],i[1])).astype(np.int).tolist() for idx,i in enumerate(shp_polar.geometry.apply(lambda x: (x.x, x.y)))}

# vals = {i:ds_sum.pcpt[:,rowcols[i][0],rowcols[i][1]].values for i in rowcols}
for community in ['Seward', 'Barrow', 'Nome','Anchorage', 'Fairbanks', 'Juneau', 'Dillingham']:
	for group in ['annualmaximum','annualmaximum-lowerlimit','annualmaximum-upperlimit']:
		# community = 'Fairbanks'
		print(community)
		print(group)
		# Atlas 14 Data:
		# atlas_fn = os.path.join(base_path,'NOAA_Atlas14','netcdf','pr_freq_noaa-atlas14_ak_partialduration.nc')
		atlas_fn = os.path.join(base_path,'NOAA_Atlas14','netcdf','pr_freq_noaa-atlas14_ak_{}_wrf_grid.nc'.format(group))
		atlas = xr.open_dataset(atlas_fn)['pr_freq']
		atlas = atlas.sel(intervals=intervals) # make them common
		ind = np.where(atlas.values == -9999)
		atlas.values[ind] = np.nan # update the out-of-bounds values in the atlas data
		# atlas_mean = pd.DataFrame({i:atlas.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}, index=durations)
		atlas_mean = pd.DataFrame({i:atlas.sel(intervals=i).isel(xc=rowcols[community][0],yc=rowcols[community][1]).values.copy() for i in intervals}, index=durations)
		atlas_mean.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pr_freq_noaa-atlas14_ak_{}_area_wide_mean_{}.csv'.format(group,community)))

		# WRF Data:
		wrf_fn = os.path.join(base_path,'wrf_pcpt','netcdf','pf_ERA-Interim_historical_ak.nc')
		wrf = xr.open_dataset(wrf_fn)['pr_freq']
		wrf.values[np.where(wrf.values == -9999)] = np.nan
		wrf = wrf.sel(durations=durations) # make them common
		# wrf_mean = pd.DataFrame({i:wrf.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}, index=durations)
		# wrf_mean.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pf_ERA-Interim_historical_area_wide_mean.csv'))
		wrf_mean = pd.DataFrame({i:wrf.sel(intervals=i).isel(xc=rowcols[community][0],yc=rowcols[community][1]).values.copy() for i in intervals}, index=durations)
		wrf_mean.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pf_ERA-Interim_historical_area_wide_mean_{}.csv'.format(community)))

		# compare
		# update the durations labels to be common between both groups.
		# atlas_mean.columns = wrf.durations.values
		diff = wrf_mean - atlas_mean
		# diff[atlas == -9999] = np.nan
		# mean = {i:diff.sel(intervals=i).values for i in intervals}
		# mean_df = pd.DataFrame(mean, index=durations)
		diff.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pf_compare_area_wide_mean_{}_{}.csv'.format(group,community)))

		if group == 'annualmaximum':
			title = 'Comparison WRF - Atlas14 Differences Area-wide Mean\nPrecip Frequency -- {}'.format(community)
		elif group == 'annualmaximum-lowerlimit':
			title = 'Comparison WRF - Atlas14 Differences Area-wide Mean\nLower Confidence Bound -- {}'.format(community)
		elif group == 'annualmaximum-upperlimit':
			title = 'Comparison WRF - Atlas14 Differences Area-wide Mean\nUpper Confidence Bound -- {}'.format(community)
		diff.plot(kind='line', title=title,figsize=(16,9))
		plt.savefig(os.path.join(base_path,'compare_atlas14_wrf','pf_compare_area_wide_mean_{}_{}.png'.format(group,community)))
		plt.close()
		plt.cla()
		
		wrf.close();atlas.close()
		del wrf, atlas, ind, atlas_mean, wrf_mean
