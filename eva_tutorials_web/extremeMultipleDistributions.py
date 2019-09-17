# eva analysis -- extremeMultipleDistributions.ipynb

import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy import stats as stats
import lmoments3 as lmoments
from lmoments3 import distr

# this is prolly unneeded
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

# data prep for the analysis
extracted_fn = '/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/pcpt_hourlies_communities_ERA-Interim_historical.csv'
df = pd.read_csv(extracted_fn, index_col=0)
cols = df.columns
df.index = pd.DatetimeIndex(df.index)
dates = pd.DataFrame([{'month':i.month,'year':i.year, 'day':i.day} for i in df.index])
dates.index = pd.DatetimeIndex(df.index)
dat = pd.concat([dates,df], axis=1)

# make an annual maximum series with the precip totals for this duration 
ams = dat.groupby('year')[cols].max() / 25.4 # make inches...

for community_name in cols:
	# Calculate moments
	# LMU = lmoments.samlmu(df) # LM-version2
	# LMU = lmoments.lmom_ratios(ams[community_name]) # LM-version3

	# Fit GEV distribution with the annual maximum series
	paras = distr.gev.lmom_fit( ams[community_name] )
	fitted_gev = distr.gev(ams[community_name])

	# get fit params based on AMS
	gev_paras = distr.gev.lmom_fit( ams[community_name] )
	exp_paras = distr.exp.lmom_fit( ams[community_name] )
	gum_paras = distr.gum.lmom_fit( ams[community_name] )
	wei_paras = distr.wei.lmom_fit( ams[community_name] )
	gpa_paras = distr.gpa.lmom_fit( ams[community_name] )
	pe3_paras = distr.pe3.lmom_fit( ams[community_name] )
	gam_paras = distr.gam.lmom_fit( ams[community_name] )
	glo_paras = distr.glo.lmom_fit( ams[community_name] )

	# set-up some return years
	# return years (1.1 to 1000)
	# T = np.array([1,2,5,10,25,50,100,200,500,1000]).astype(np.float64)
	T = np.arange(0.1, 999.1, 0.1) + 1

	# fit the distribution
	fitted_gev = distr.gev(**gev_paras)
	fitted_exp = distr.exp(**exp_paras)
	fitted_gum = distr.gum(**gum_paras)
	fitted_wei = distr.wei(**wei_paras)
	fitted_gpa = distr.gpa(**gpa_paras)
	fitted_pe3 = distr.pe3(**pe3_paras)
	fitted_gam = distr.gam(**gam_paras)
	fitted_glo = distr.glo(**glo_paras)


	# fit_obj_dict = {'gev':fitted_gev,'exp':fitted_exp,'gum':fitted_gum,
	# 				'wei':fitted_wei,'gpa':fitted_gpa,'pe3':fitted_pe3,
	# 				'gam':fitted_gam,'glo':fitted_glo,}

	# get extreme precip at desired return years
	gevST = fitted_gev.ppf(1.0-1./T)
	expST = fitted_exp.ppf(1.0-1./T)
	gumST = fitted_gum.ppf(1.0-1./T)
	weiST = fitted_wei.ppf(1.0-1./T)
	gpaST = fitted_gpa.ppf(1.0-1./T)
	pe3ST = fitted_pe3.ppf(1.0-1./T)
	gamST = fitted_gam.ppf(1.0-1./T)
	gloST = fitted_glo.ppf(1.0-1./T)

	# setup plotting parameters
	plt.xscale('log')
	plt.xlabel('Average Return Interval (Year)')
	plt.ylabel('Precipitation (inches)')

	# draw extreme values from GEV distribution
	line1, = plt.plot(T, gevST, 'g', label='GEV')
	line2, = plt.plot(T, expST, 'r', label='EXP')
	line3, = plt.plot(T, gumST, 'b', label='GUM')
	line4, = plt.plot(T, weiST, 'y', label='WEI')
	line5, = plt.plot(T, gpaST, 'c', label='GPA')
	line6, = plt.plot(T, pe3ST, 'm', label='PE3')
	line7, = plt.plot(T, gamST, 'k', label='GAM')
	line8, = plt.plot(T, gloST, c='orange', label='GLO')

	# draw extreme values from observations(empirical distribution)
	N    = np.r_[1:len(ams[community_name].index)+1]*1.0 #must *1.0 to convert int to float
	Nmax = max(N)

	plt.scatter(Nmax/N, sorted(ams[community_name])[::-1], color = 'orangered', facecolors='none', label='Empirical')
	plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

	plt.savefig('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/extremeMultipleDistributions/{}_distributions_all_fitted.png'.format(community_name))
	plt.close()


	# select an optimal distribution: 
	P0   = (N-1.)/Nmax
	P    = np.delete(P0,0)

	obs = sorted(ams[community_name])[1:]

	# get extreme precip at desired probabilities
	gevSTo = fitted_gev.ppf(P)
	expSTo = fitted_exp.ppf(P)
	gumSTo = fitted_gum.ppf(P)
	weiSTo = fitted_wei.ppf(P)
	gpaSTo = fitted_gpa.ppf(P)
	pe3STo = fitted_pe3.ppf(P)
	gamSTo = fitted_gam.ppf(P)
	gloSTo = fitted_glo.ppf(P)



	# get some fitness stats
	fit_dict = {'gev':gevSTo, 'exp':expSTo, 'gum':gumSTo, 'wei':weiSTo, 'gpa':gpaSTo, 'pe3':pe3STo, 'gam':gamSTo, 'glo':gloSTo, }
	ks_df = pd.DataFrame({i:stats.ks_2samp(obs,fit_dict[i]).__dict__ for i in fit_dict})
	ks_df.to_csv('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/extremeMultipleDistributions/{}_ks_statistics.csv'.format(community_name))
	
	# # do ks test
	# ks = [('GEV', stats.ks_2samp(obs, gevSTo)), ('EXP', stats.ks_2samp(obs, expSTo)),
	#       ('GUM', stats.ks_2samp(obs, gumSTo)), ('WEI', stats.ks_2samp(obs, weiSTo)),
	#       ('GPA', stats.ks_2samp(obs, gpaSTo)), ('PE3', stats.ks_2samp(obs, pe3STo)), 
	#       ('GAM', stats.ks_2samp(obs, gamSTo)), ('GLO', stats.ks_2samp(obs, gloSTo))]

	break

	# # put em in a data frame for visualization of the vals.
	# labels = ['Distribution', 'KS (statistics, pvalue)']
	# pd.DataFrame(ks, columns=labels).to_csv('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/extremeMultipleDistributions/{}_ks_statistics.csv'.format(community_name))

	# plot the GEV:
	# ks_df.to_csv('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/extremeMultipleDistributions/{}_ks_statistics.csv'.format(community_name))
	ks_df.plot(logx=True)

	# plot the GEV
	plt.xscale('log')
	plt.xlabel('Average Return Interval (Year)')
	plt.ylabel('Precipitation (inches)')
	# line1, = plt.plot(T, gevST, 'g', label='GEV')

	plt.scatter(Nmax/N, sorted(ams[community_name])[::-1], color = 'orangered', facecolors='none', label='Empirical')
	plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

	plt.savefig('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/extremeMultipleDistributions/{}_gev_obs_fitted.png'.format(community_name))
	plt.close()