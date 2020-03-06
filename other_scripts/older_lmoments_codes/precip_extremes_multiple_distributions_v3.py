# LOAD MODULES
import os, warnings, io
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import lmoments3 as lmoments
from lmoments3 import distr
import scikits.bootstrap
import requests

# numpy interpreter temporary config
np.set_printoptions(suppress=True)


# read data from an online resource
base_url = 'https://www.snap.uaf.edu/webshared/Michael/data'
url = base_url+'/pcpt_day_sum_communities_v2_ERA-Interim_historical.csv' 
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')), index_col=0)
df.index = pd.DatetimeIndex(df.index) # make sure index is pd.DatetimeIndex
df.head(5) # show first 5 rows

#@title Select community and duration { run: "auto" }
community = "Fairbanks" #@param ['Anchorage', 'Barrow', 'Dillingham', 'Fairbanks', 'Juneau', 'Nome', 'Seward']
duration = "24h" #@param ['24h','2d','3d','4d','7d','10d','20d','30d','45d','60d']
#@markdown

"""## Compute Duration Aggregation and Generate Annual Maximum Series (AMS)
---
The annual maximum series is the root dataset used in all EVA computations.  This (i think) is what they did in the NOAA Atlas 14, any guidance here (or on any of these steps) is desired and welcomed.
"""

# subset the DataFrame to the community chosen
comm_df = df[community]

# make any trace precip that is less than a hundredth of a mm ZERO
comm_df[comm_df < .01] = 0

# make a duration series -- SUM
dur = comm_df.resample(duration).sum()

# make an annual maximum series
ams = dur.resample('Y').max()
ams.index = ams.index.map(lambda x: x.year)
ams = ams*0.0393701


# """## Plot the AMS"""

# title = 'Annual Maximum Series\n{} - duration: {}'\
#           .format(community, duration)
# ax = ams.plot(kind='bar', figsize=(16,5), color=['grey'], title=title)

# SET UP RETURN YEARS.  [THIS IS SOMETHING I AM NOT FEELING CONFIDENT ABOUT.]
# T = np.array([2,5,10,25,50,100,200,500,1000]).astype(np.float64)
# T = np.arange(len(ams))+2
T = np.arange(0.1, 1000.0, 0.1) + 1

# build a dictionary of the distribution functions (from lmoments3)
dist_types = {'gev':distr.gev,'glo':distr.glo,'gno':distr.gno,
             'gpa':distr.gpa,'gum':distr.gum,'kap':distr.kap,
             'nor':distr.nor,'pe3':distr.pe3,'wei':distr.wei,}

# generate the parameters for the distribution based on the lmoments
dist_params = {k:dist_types[k].lmom_fit(ams) for k in dist_types}

with warnings.catch_warnings(record=True):
  # make the frozen distribution objects using the above dicts
  frozen_dist_objs = {k:dist_types[k](**dist_params[k]) for k in dist_params}

  intervals = {k:frozen_dist_objs[k].ppf(1.0-1./T) for k in frozen_dist_objs}
  intervals_df = pd.DataFrame(intervals, index=T)
  
intervals_df.head() # show the first 5 rows of the newly generated dataframe

# """### VISUALLY COMPARE WITH THE EMPIRICAL DISTRIBUTION"""

# # plot using pandas
# title='Comparison of Distributions\n{}, Alaska {} Duration'.format(community, duration)
# intervals_df.plot(logx=True, figsize=(16,5), title=title)

# # draw the observed AMS values as points
# N    = np.r_[1:len(ams)+1]*1.0 #must *1.0 to convert int to float
# Nmax = max(N)
# ams = ams
# plt.scatter(Nmax/N, sorted(ams)[::-1], 
#             color = 'black', facecolors='black', label='Empirical',zorder=10)

# # update some elements
# plt.xlabel('Average Return Interval (Year)')
# plt.ylabel('Precipitation (inches)')
# plt.show()


# P0   = (N-1.)/Nmax
# P    = np.delete(P0,0)

# obs = sorted(ams)
# new_dat = {k:frozen_dist_objs[k].ppf(1-(1.0/T)) for k in frozen_dist_objs}

# # display the output fit statistics
# # out_stats = {k:tuple(stats.ks_2samp(obs, new_dat[k])) for k in new_dat}
# out_stats = {k:tuple(np.array(stats.anderson_ksamp([ams, sorted(new_dat[k])]))[[0,2]]) for k in new_dat}
# stat_df = pd.DataFrame(out_stats, index=['statistic','significance_level']).T
# stat_df.sort_values(['statistic','significance_level'])

"""## Select one of the distributions examined"""

#@title Select a distribution { run: "auto" }
distribution = "gev" #@param ['nor','gpa','wak','wei','kap','pe3','gno','gev','gum','glo']
#@markdown


# title='{} Distribution\n{}, Alaska {} Duration'.format(distribution.upper(), community, duration)
# intervals_df[distribution].plot(kind='line', logx=True, 
#                                 color='darkgreen', figsize=(16,5), 
#                                 title=title)

# # draw extreme values from observations(empirical distribution)
# N    = np.r_[1:len(ams.index)+1]*1.0 #must *1.0 to convert int to float
# Nmax = max(N)

# plt.scatter(Nmax/N, sorted(ams)[::-1], color = 'black', 
#             facecolors='black', label='Empirical', zorder=2)

# plt.show()


# # # # # # # # # # # # # # # # # # # # 
# EXPERIMENTAL BOOTSTRAPPING PROCEDURE
# # # # # # # # # # # # # # # # # # # # 

from collections import OrderedDict
import scikits.bootstrap as boot

def bootstrap_ci(data, statfunction=np.average, alpha = 0.05, 
                 n_samples = 100):
    """
    Given a set of data ``data``, and a statistics function ``statfunction`` that
    applies to that data, computes the bootstrap confidence interval for
    ``statfunction`` on that data. Data points are assumed to be delineated by
    axis 0.
    
    This function has been derived and simplified from scikits-bootstrap 
    package created by cgevans (https://github.com/cgevans/scikits-bootstrap).
    All the credits shall go to him.
    **Parameters**
    
    data : array_like, shape (N, ...) OR tuple of array_like all with shape (N, ...)
        Input data. Data points are assumed to be delineated by axis 0. Beyond this,
        the shape doesn't matter, so long as ``statfunction`` can be applied to the
        array. If a tuple of array_likes is passed, then samples from each array (along
        axis 0) are passed in order as separate parameters to the statfunction. The
        type of data (single array or tuple of arrays) can be explicitly specified
        by the multi parameter.
    statfunction : function (data, weights = (weights, optional)) -> value
        This function should accept samples of data from ``data``. It is applied
        to these samples individually. 
    alpha : float, optional
        The percentiles to use for the confidence interval (default=0.05). The 
        returned values are (alpha/2, 1-alpha/2) percentile confidence
        intervals. 
    n_samples : int or float, optional
        The number of bootstrap samples to use (default=100)
        
    **Returns**
    
    confidences : tuple of floats
        The confidence percentiles specified by alpha
    **Calculation Methods**
    
    'pi' : Percentile Interval (Efron 13.3)
        The percentile interval method simply returns the 100*alphath bootstrap
        sample's values for the statistic. This is an extremely simple method of 
        confidence interval calculation. However, it has several disadvantages 
        compared to the bias-corrected accelerated method.
        
        If you want to use more complex calculation methods, please, see
        `scikits-bootstrap package 
        <https://github.com/cgevans/scikits-bootstrap>`_.
    **References**
    
        Efron (1993): 'An Introduction to the Bootstrap', Chapman & Hall.
    """

    def bootstrap_indexes(data, n_samples=10000):
        """
    Given data points data, where axis 0 is considered to delineate points, return
    an generator for sets of bootstrap indexes. This can be used as a list
    of bootstrap indexes (with list(bootstrap_indexes(data))) as well.
        """
        for _ in range(n_samples):
            yield _randint(data.shape[0], size=(data.shape[0],))    
    
    alphas = np.array([alpha / 2,1 - alpha / 2])

    data = np.array(data)
    tdata = (data,)
    
    # We don't need to generate actual samples; that would take more memory.
    # Instead, we can generate just the indexes, and then apply the statfun
    # to those indexes.
    bootindexes = bootstrap_indexes(tdata[0], n_samples)
    stat = np.array([statfunction(*(x[indexes] for x in tdata)) for indexes in bootindexes])
    stat.sort(axis=0)

    # Percentile Interval Method
    avals = alphas

    nvals = np.round((n_samples - 1)*avals).astype('int')

    if np.any(nvals == 0) or np.any(nvals == n_samples - 1):
        _warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
    elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
        _warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

    if nvals.ndim == 1:
        # All nvals are the same. Simple broadcasting
        return stat[nvals]
    else:
        # Nvals are different for each data point. Not simple broadcasting.
        # Each set of nvals along axis 0 corresponds to the data at the same
        # point in other axes.
        return stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]


NSAMPLES=50000
def run_bootstrap(dat, lmom_fitted):
    ''' 
    Calculate confidence intervals using parametric bootstrap and the
    percentil interval method
    This is used to obtain confidence intervals for the estimators and
    the return values for several return values.    
    More info about bootstrapping can be found on:
        - https://github.com/cgevans/scikits-bootstrap
        - Efron: "An Introduction to the Bootstrap", Chapman & Hall (1993)
        - https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

    parametric bootstrap for return levels and parameters   
    '''
    
    # function to bootstrap
    def sample_return_intervals(data):
        # get a random sampling of the fitted distribution
        sample = lmom_fitted.rvs(len(dat))
        # return the fitted params of this new randomized sample        
        # paras = distr.gev.lmom_fit(sample)
        paras = dist_types[distribution].lmom_fit(sample) # watch this it can get you into trouble
        
        samplefit = [ paras[i] for i in ['loc', 'scale', 'c']]
        sample_fitted = dist_types[distribution](**paras)

        # set up the return intervals to pull from fitted distribution
        # intervals = np.arange(0.1, 1000.0, 0.1) + 1 
        intervals = T
        sample_intervals = sample_fitted.ppf(1.0-1./intervals)
        # this basically just puts the fitted params at the START of the series
        # they are subsequently sliced out in the wrapper function.
        res = samplefit
        res.extend(sample_intervals.tolist())
        return tuple(res)

    # the calculations itself
    out = boot.ci(dat, statfunction=sample_return_intervals, \
                        alpha=0.05, n_samples=NSAMPLES, \
                        method='bca', output='lowhigh')
    
    
    ci_Td = out[0, 3:]
    ci_Tu = out[1, 3:]
    params_ci = OrderedDict()
    params_ci['c']    = (out[0,0], out[1,0])
    params_ci['loc'] = (out[0,1], out[1,1])
    params_ci['scale']    = (out[0,2], out[1,3])
    
    return {'ci_Td':ci_Td, 'ci_Tu':ci_Tu, 'params_ci':params_ci}

# print(distribution)
# lmom_fitted = frozen_dist_objs[distribution]
# paras = dist_types[distribution].lmom_fit(sample) # watch this it can get you into trouble
# print(paras)
# print(dist_params[distribution])
# # sample = lmom_fitted.rvs(len(ams))
# # lmom_fitted.rvs

# get confidence intervals using a bootstrapping procedure
bootout = run_bootstrap(ams, frozen_dist_objs[distribution])

"""## Plot the Estimated Confidence Bounds with the AMS observed series"""

# breakup the outputs
ci_Td     = bootout["ci_Td"]
ci_Tu     = bootout["ci_Tu"]
params_ci = bootout["params_ci"]

# plot it
fig, ax = plt.subplots(figsize=(16, 5))
plt.setp(ax.lines, linewidth=2, color='magenta')

ax.set_title("{} Distribution".format(distribution))
ax.set_xlabel("Return Period (Year)")
ax.set_ylabel("Precipitation")
ax.semilogx(T, intervals[distribution])
ax.scatter(Nmax/N, sorted(ams)[::-1], color='orangered')

ax.semilogx(T, ci_Td, '--')
ax.semilogx(T, ci_Tu, '--')
ax.fill_between(T, ci_Td, ci_Tu, color='0.75', alpha=0.5)
plt.show()

# distr.gam.lmom_fit?
paras?

