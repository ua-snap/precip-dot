# NOTES:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html
# pip install lmoments3
# https://lmoments3.readthedocs.io/en/stable/#l-moment-estimation-from-sample-data
# THE ONE BELOW MIGHT BE REALLY IMPORTANT:
# https://www.linkedin.com/pulse/beginners-guide-carry-out-extreme-value-analysis-2-chonghua-yin/


if __name__ == '__main__':
	import xarray as xr
	import lmoments3 as lm
	import lmoments3 as lm
	from lmoments3 import distr

	ds = xr.open_dataset('/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/PCPT_wrf_hourly_gfdl_rcp85_2020_1D_ams.nc')
	data = ds.PCPT[:, 130,130]
	lm.lmom_ratios(data, nmom=5)

	# Fitting distribution functions to sample data
	paras = distr.gam.lmom_fit(data)
	fitted_gam = distr.gam(**paras)
	median = fitted_gam.ppf(0.5)