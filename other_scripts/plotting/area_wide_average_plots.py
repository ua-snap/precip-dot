import os, matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

base_path = '/Users/malindgren/Documents/TEMP/compare_atlas14_wrf'
community = 'Fairbanks'
names = ['pf_compare_area_wide_mean_{}.csv'.format(community), 'pf-upperlimit_compare_area_wide_mean_{}.csv'.format(community), 'pf-lowerlimit_compare_area_wide_mean_{}.csv'.format(community)]


# read in the data
data = { i:pd.read_csv(os.path.join(base_path, i), index_col=0) for i in names }

# data['pf-upperlimit_compare_area_wide_mean.csv'].plot(kind='line')
# plt.savefig('/Users/malindgren/Documents/TEMP/compare_atlas14_wrf_TEST.png')
# plt.close()

pd.DataFrame([data[d]['100'] for d in data]).T.plot()
plt.savefig('/Users/malindgren/Documents/TEMP/compare_atlas14_wrf_TEST.png')
plt.close()


# JUST ATLAS14

import os, matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

base_path = '/Users/malindgren/Documents/TEMP/compare_atlas14_wrf'
names = ['pr_freq_noaa-atlas14_ak_annualmaximum-lowerlimit_area_wide_mean.csv','pr_freq_noaa-atlas14_ak_annualmaximum-upperlimit_area_wide_mean.csv','pr_freq_noaa-atlas14_ak_annualmaximum_area_wide_mean.csv']
for i in names:
	pd.read_csv(os.path.join(base_path, i), index_col=0).plot(kind='line',figsize=(16,9))
	plt.savefig('/Users/malindgren/Documents/TEMP/{}'.format(i.replace('.csv','.png')))
	plt.close()


# JUST WRF
base_path = '/Users/malindgren/Documents/TEMP/compare_atlas14_wrf'
names = ['pf_upper-ci_ERA-Interim_historical_area_wide_mean.csv','pf_lower-ci_ERA-Interim_historical_area_wide_mean.csv','pf_ERA-Interim_historical_area_wide_mean.csv']
for i in names:
	pd.read_csv(os.path.join(base_path, i), index_col=0).plot(kind='line',figsize=(16,9))
	plt.savefig('/Users/malindgren/Documents/TEMP/{}'.format(i.replace('.csv','.png')))
	plt.close()


