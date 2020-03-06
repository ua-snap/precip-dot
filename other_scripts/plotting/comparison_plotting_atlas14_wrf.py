# comparison plotting
import os, glob
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

# some setup
base_path = '/workspace/Shared/Tech_Projects/DOT/project_data'
durations = ['60m','2h','3h','6h','12h','24h','3d','4d','7d','10d','20d','30d','45d','60d',] #'2d'
intervals = [2,5,10,25,50,100,200,500,1000] # 1

# Atlas 14 Data:
# atlas_fn = os.path.join(base_path,'NOAA_Atlas14','netcdf','pr_freq_noaa-atlas14_ak_partialduration.nc')
atlas_fn = os.path.join(base_path,'NOAA_Atlas14','netcdf','pr_freq_noaa-atlas14_ak_annualmaximum_wrf_grid.nc')

atlas = xr.open_dataset(atlas_fn)['pr_freq']
atlas = atlas.sel(intervals=intervals) # make them common
ind = np.where(atlas.values == -9999)
atlas.values[ind] = np.nan # update the out-of-bounds values in the atlas data
atlas_mean = pd.DataFrame({i:atlas.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}, index=durations)
atlas_mean.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pr_freq_noaa-atlas14_ak_annualmaximum_area_wide_mean.csv'))


# WRF Data:
wrf_fn = os.path.join(base_path,'wrf_pcpt','netcdf','pf_ERA-Interim_historical_ak.nc')
wrf = xr.open_dataset(wrf_fn)['pr_freq']
wrf.values[np.where(wrf.values == -9999)] = np.nan
wrf = wrf.sel(durations=durations) # make them common
wrf_mean = pd.DataFrame({i:wrf.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}, index=durations)
wrf_mean.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pf_ERA-Interim_historical_area_wide_mean.csv'))

# compare
# update the durations labels to be common between both groups.
atlas['durations'].values = wrf.durations.values
diff = wrf - atlas
# diff[atlas == -9999] = np.nan
mean = {i:diff.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}
mean_df = pd.DataFrame(mean, index=durations)
mean_df.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pf_compare_area_wide_mean.csv'))
mean_df.plot(kind='line', title='Comparison WRF - Atlas14 Differences Area-wide Mean\nPrecip Frequency',figsize=(16,9))
plt.savefig(os.path.join(base_path,'compare_atlas14_wrf','pf_compare_area_wide_mean.png'))
plt.close()
plt.cla()


# # # # LOWER LIMIT

# comparison plotting
import os, glob
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

# some setup
base_path = '/workspace/Shared/Tech_Projects/DOT/project_data'
durations = ['60m','2h','3h','6h','12h','24h','3d','4d','7d','10d','20d','30d','45d','60d',] #'2d'
intervals = [2,5,10,25,50,100,200,500,1000] # 1

# Atlas 14 Data:
# atlas_fn = os.path.join(base_path,'NOAA_Atlas14','netcdf','pr_freq_noaa-atlas14_ak_partialduration.nc')
atlas_fn = os.path.join(base_path,'NOAA_Atlas14','netcdf','pr_freq_noaa-atlas14_ak_annualmaximum-lowerlimit_wrf_grid.nc')

atlas = xr.open_dataset(atlas_fn)['pr_freq']
atlas = atlas.sel(intervals=intervals) # make them common
ind = np.where(atlas.values == -9999)
atlas.values[ind] = np.nan # update the out-of-bounds values in the atlas data
atlas_mean = pd.DataFrame({i:atlas.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}, index=durations)
atlas_mean.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pr_freq_noaa-atlas14_ak_annualmaximum-lowerlimit_area_wide_mean.csv'))

# WRF Data:
wrf_fn = os.path.join(base_path,'wrf_pcpt','netcdf','pf_lower-ci_ERA-Interim_historical_ak.nc')
wrf = xr.open_dataset(wrf_fn)['pr_freq']
wrf.values[np.where(wrf.values == -9999)] = np.nan
wrf = wrf.sel(durations=durations) # make them common
wrf_mean = pd.DataFrame({i:wrf.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}, index=durations)
wrf_mean.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pf_lower-ci_ERA-Interim_historical_area_wide_mean.csv'))


# compare
# update the durations labels to be common between both groups.
atlas['durations'].values = wrf.durations.values
diff = wrf - atlas
# diff[atlas == -9999] = np.nan
mean = {i:diff.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}
mean_df = pd.DataFrame(mean, index=durations)
mean_df.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pf-lowerlimit_compare_area_wide_mean.csv'))
mean_df.plot(kind='line', title='Comparison WRF - Atlas14 Differences Area-wide Mean\nLower Confidence Bound',figsize=(16,9))
plt.savefig(os.path.join(base_path,'compare_atlas14_wrf','pf-lowerlimit_compare_area_wide_mean.png'))
plt.close()
plt.cla()


# # # # UPPER-LIMIT
# comparison plotting
import os, glob
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

# some setup
base_path = '/workspace/Shared/Tech_Projects/DOT/project_data'
durations = ['60m','2h','3h','6h','12h','24h','3d','4d','7d','10d','20d','30d','45d','60d',] #'2d'
intervals = [2,5,10,25,50,100,200,500,1000] # 1

# Atlas 14 Data:
# atlas_fn = os.path.join(base_path,'NOAA_Atlas14','netcdf','pr_freq_noaa-atlas14_ak_partialduration.nc')
atlas_fn = os.path.join(base_path,'NOAA_Atlas14','netcdf','pr_freq_noaa-atlas14_ak_annualmaximum-upperlimit_wrf_grid.nc')

atlas = xr.open_dataset(atlas_fn)['pr_freq']
atlas = atlas.sel(intervals=intervals) # make them common
# ind = np.where(atlas.values == -9999)
atlas.values[ind] = np.nan # update the out-of-bounds values in the atlas data
atlas_mean = pd.DataFrame({i:atlas.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}, index=durations)
atlas_mean.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pr_freq_noaa-atlas14_ak_annualmaximum-upperlimit_area_wide_mean.csv'))


# WRF Data:
wrf_fn = os.path.join(base_path,'wrf_pcpt','netcdf','pf_upper-ci_ERA-Interim_historical_ak.nc')
wrf = xr.open_dataset(wrf_fn)['pr_freq']
wrf.values[np.where(wrf.values == -9999)] = np.nan
wrf = wrf.sel(durations=durations) # make them common
wrf_mean = pd.DataFrame({i:wrf.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}, index=durations)
wrf_mean.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pf_upper-ci_ERA-Interim_historical_area_wide_mean.csv'))

# compare
# update the durations labels to be common between both groups.
atlas['durations'].values = wrf.durations.values
diff = wrf - atlas
# diff[atlas == -9999] = np.nan
mean = {i:diff.sel(intervals=i).mean(axis=(1,2)).values for i in intervals}
mean_df = pd.DataFrame(mean, index=durations)
mean_df.to_csv(os.path.join(base_path,'compare_atlas14_wrf','pf-upperlimit_compare_area_wide_mean.csv'))
mean_df.plot(kind='line', title='Comparison WRF - Atlas14 Differences Area-wide Mean\nUpper Confidence Bound', figsize=(16,9))
plt.savefig(os.path.join(base_path,'compare_atlas14_wrf','pf-upperlimit_compare_area_wide_mean.png'))
plt.close()
plt.cla()
