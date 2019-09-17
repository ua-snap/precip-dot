def run_bootstrap(dat, lmom_fitted):
    # Calculate confidence intervals using parametric bootstrap and the
    # percentil interval method
    # This is used to obtain confidence intervals for the estimators and
    # the return values for several return values.    
    # More info about bootstrapping can be found on:
    #     - https://github.com/cgevans/scikits-bootstrap
    #     - Efron: "An Introduction to the Bootstrap", Chapman & Hall (1993)
    #     - https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

    # parametric bootstrap for return levels and parameters   

    # The function to bootstrap     
    def sample_return_intervals(data):
        # get a random sampling of the fitted distribution
        sample = lmom_fitted.rvs(len(dat))
        # return the fitted params of this new randomized sample        
        paras = distr.gev.lmom_fit(sample)
        samplefit = [ paras[i] for i in ['loc', 'scale', 'c']]
        sample_fitted = distr.gev(**paras)

        # set up the return intervals to pull from fitted distribution
        intervals = np.arange(0.1, 999.1, 0.1) + 1
        sample_intervals = sample_fitted.ppf(1.0-1./intervals)
        res = samplefit
        res.extend(sample_intervals.tolist())
        return tuple(res)

    # the calculations itself
    out = boot.ci(dat, statfunction=sample_return_intervals, 
                        alpha=0.05, n_samples=500, 
                        method='pct', output='lowhigh')

    ci_Td = out[0, 3:]
    ci_Tu = out[1, 3:]
    params_ci = OrderedDict()
    params_ci['c']    = (out[0,0], out[1,0])
    params_ci['loc'] = (out[0,1], out[1,1])
    params_ci['scale']    = (out[0,2], out[1,3])
    
    return {'ci_Td':ci_Td, 'ci_Tu':ci_Tu, 'params_ci':params_ci}

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    import os
    import pandas as pd
    from lmoments3 import distr
    from collections import OrderedDict
    import scikits.bootstrap as boot

    # # # R VERSION BLOG: # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # http://blogs2.datall-analyse.nl/2016/02/17/extreme_value_analysis_maxima/#more-120
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # # On import, make sure that InstabilityWarnings are not filtered out.
    # _warnings.simplefilter('always', InstabilityWarning)
    # _warnings.simplefilter('always', UserWarning)

    # rcParams['figure.figsize'] = 15, 6

    path = '/workspace/UA/malindgren/repos/A-Beginner-Guide-to-Carry-out-Extreme-Value-Analysis-with-Codes-in-Python'
    os.chdir(path)

    data = pd.read_csv('./fortprec.txt', sep ='\t')
    df = data.groupby("year").Prec.max()

    avi = np.array([1,2,5,10,25,50,100,200,500,1000])
    gevfit = distr.gev.lmom_fit(df)
    fitted_gev = distr.gev(**gevfit)

    T  = np.arange(0.1, 999.1, 0.1) + 1
    sT = fitted_gev.ppf( 1.0-(1.0 / T))

    # prepare index for obs
    N    = np.r_[1:len(df.index)+1]*1.0 # must *1.0 to convert int to float
    Nmax = max(N)

    # get confidence intervals
    bootout = boot.ci(df, statfunction=func, alpha=0.05, n_samples=10000,
                        method='bca', output='lowhigh', multi=None)

    # breakup the outputs
    ci_Td     = bootout["ci_Td"]
    ci_Tu     = bootout["ci_Tu"]
    params_ci = bootout["params_ci"]

    # plot it out:
    fig, ax = plt.subplots(figsize=(16, 5))
    plt.setp(ax.lines, linewidth=2, color='magenta')

    ax.set_title("GEV Distribution")
    ax.set_xlabel("Return Period (Year)")
    ax.set_ylabel("Precipitation")
    ax.semilogx(T, sT)
    ax.scatter(Nmax/N, sorted(df)[::-1], color='orangered')

    ax.semilogx(T, ci_Td, '--')
    ax.semilogx(T, ci_Tu, '--')
    ax.fill_between(T, ci_Td, ci_Tu, color='0.75', alpha=0.5)

    plt.savefig('/workspace/Shared/Tech_Projects/DOT/project_data/testing_eva/test_confInterval.png')
    plt.close()
    plt.cla()

