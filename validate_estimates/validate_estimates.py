# Plot differences between WRF-Interim-derived PF estimates produced by Neptune and 
#   current NOAA Atlas 14 estimates

# read NOAA Atlas 14 files and combine all estimates (pf, CI upper bound, CI lower bound) 
#   into one DataFrame
def read_a14(fp):
    # omitting first column (is the duration column)
    cols = [i for i in range(11)]
    del cols[1]
    pf = pd.read_csv(fp, skiprows=18, nrows=15, header=None, index_col=0, usecols=cols)
    pf_upper = pd.read_csv(fp, skiprows=40, nrows=15, header=None, index_col=0, usecols=cols)
    pf_lower = pd.read_csv(fp, skiprows=62, nrows=15, header=None, index_col=0, usecols=cols)
    pf = pd.concat([pf, pf_upper, pf_lower], axis=1)
    return pf

# extract all WRF-based estimates (pf, CI upper bound, CI lower bound) by WGS84 coordinates
def extr_wrf(wrf_ds, coords):
    # return index of arr value nearest to k
    def where(k, arr):
        ad = abs(arr - k)
        return np.where(ad == min(ad))[0]
    # apply a function to each location to extract WRF-based estimates 
    wrf_proj = wrf_ds.xc.attrs['proj_parameters']
    in_proj = Proj('+init=epsg:4326')
    out_proj = Proj(wrf_proj)
    x, y = transform(in_proj, out_proj, coords[0], coords[1])
    xc = wrf_ds.xc.values
    yc = wrf_ds.yc.values
    idx, idy = where(x, xc), where(y, yc)
    # extract and combine WRF-based estimates
    pf = wrf_ds['pf'][:, idx, idy].values.flatten()
    pf_upper = wrf_ds['pf-upper'][:, idx, idy].values.flatten()
    pf_lower = wrf_ds['pf-lower'][:, idx, idy].values.flatten()
    pf = np.array([pf, pf_upper, pf_lower]).flatten()
    return pf

# save plots of deltas between Atlas 14 estimates and WRF-based estimates for all
#   return intervals, for all durations
def plot_deltas(wrf_dir, val_dir, wrf_mod): 
    tic = time.clock()
    print('Generating plots of deltas for', wrf_mod, '...')
    # prep static data (for appending w/ deltas)
    estimate_type = ['pf'] * 9, ['pf_upper'] * 9, ['pf_lower'] * 9
    estimate_type = [item for sublist in estimate_type for item in sublist]
    estimate_type = pd.Series(estimate_type, dtype='category')    
    ri = ['2', '5', '10', '25', '50', '100', '200', '500', '1000']
    ri = pd.Series(np.tile(ri, 3), dtype='category')
    # AK locations with GPS coordinates
    a14_dir = os.path.join(val_dir, 'atlas14_ak_samples')
    locations = pd.read_csv(os.path.join(a14_dir, 'locations.csv'))
    # for every ERA-Interim-based file,
    # iterate over durations
    durations = ['60m', '2h', '3h', '6h', '12h', '24h', '2d', '3d', '4d', '7d', '10d', '20d', '30d', '45d', '60d']
    # dict for extracting Atlas 14 by duration
    a14_dur = {'60m': '60-min:', '2h': '2-hr:', '3h': '3-hr:', '6h': '6-hr:', '12h': '12-hr:',
               '24h': '24-hr:', '2d': '2-day:', '3d':'3-day:', '4d': '4-day:', '7d': '7-day:', 
               '10d': '10-day:', '20d': '20-day:', '30d': '30-day:', '45d': '45-day:', '60d': '60-day:'}
    wrf_fn = 'pcpt_{}_historical_sum_wrf_{}_1979-2015_intervals.nc'
    # build dict of all Atlas 14 PF estimates
    a14_fns = os.listdir(a14_dir)
    a14_fns = [e for e in a14_fns if e.endswith('_pf.csv')]
    stids = []
    a14_data = {}
    for a14_fn in a14_fns:
        stid = a14_fn[:4]
        stids.append(stid)
        a14_data[stid] = read_a14(os.path.join(a14_dir, a14_fn))
    # loop over durations (WRF-based files)
    for duration in durations:
        # if duration == '2h':
        #     break
        # init df for plots, first column for type of estimate (pf, pf_upper, pf_lower)
        plt_df = pd.DataFrame()
        wrf = xr.open_dataset(os.path.join(wrf_dir, wrf_fn.format(wrf_mod, duration))).load()
        for stid in stids:
            # get stid, coordinates
            # stid = a14_fn[:4]
            coords = locations[['lon', 'lat']][locations['stid'] == stid].values.flatten()
            # get NOAA Atlas 14 estimates for 2yr to 1000yr for particular site
            #fp = os.path.join(a14_dir, a14_fn)
            #a14_pf = read_a14(fp, i)
            a14_pf = a14_data[stid].loc[a14_dur[duration]].values
            # get WRF-based estimates and 
            wrf_pf = extr_wrf(wrf, coords)
            deltas = a14_pf - wrf_pf
            stid = pd.Series(np.repeat(stid, 27), dtype='category')
            data = {'stid': stid, 'ri': ri, 'estimate_type': estimate_type, 'delta': deltas}
            #data = {'stid': stid, 'ri': ri, 'a14_pf': a14_pf, 'wrf_pf': pf}
            plt_df = plt_df.append(pd.DataFrame(data))
        # plot deltas vs Return Interval
        sns.set(style='ticks', color_codes=True, font_scale=0.8)
        g = sns.FacetGrid(plt_df, col='stid', hue='estimate_type', col_wrap=5, 
                          hue_order=['pf_upper', 'pf', 'pf_lower'], 
                          hue_kws=dict(marker=['^', '.', 'v']))
        g.map(plt.axhline, y=0, ls=":", c=".5")
        g.map(plt.scatter, "ri", "delta").set_titles('{col_name}')
        g.add_legend()
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle('Duration: ' + duration, fontsize=18)
        g.savefig("{}/plots/deltas_{}_{}.png".format(val_dir, wrf_mod, duration))
        plt.close()
        print(duration, 'saved')
    print(wrf_mod, 'completed,', round(time.clock() - tic, 2))

# embed validation plots in html
def write_html(val_dir):
    print('Saving plots to summary.html', end='... ')
    plots_dir = os.path.join(val_dir, 'plots')
    plot_fns = os.listdir(plots_dir)
    wrf_mods = ['ERA-Interim', 'NCAR-CCSM4', 'GFDL-CM3']
    # collect tags for all plot images
    img_tags = []
    for mod in wrf_mods:
        mod_fns = [fn for fn in plot_fns if (mod in fn)]
        for fn in mod_fns:
            plot_fp = os.path.join(plots_dir, fn)
            data_uri = base64.b64encode(open(plot_fp, 'rb').read()).decode('utf-8')
            img_tags.append('<img src="data:image/png;base64,{}">'.format(data_uri))
    fp = os.path.join(val_dir, 'summary.html')
    f = open(fp,'w')
    message = """<html>
    <head></head>
    <body>
    <h1> Validate WRF-Based PF Estimates</h1>
    <p>This validation effort uses precipitation frequency (PF) estimates from the \
    NOAA Atlas 14 webtool for 10 Alaskan locations to check the WRF-based PF \
    estimates being produced. The following plots depict the 'deltas' of estimates \
    between these two sources, computed as the Atlas 14 estimate minus the WRF-based estimate. This is done for \
    all return intervals between 2y and 1000y. 
    </p>
    <h2>ERA-Interim (1979-2015)</h1>
    <p>{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}</p>
    <h2>NCAR-CCSM4 (historical, 1979-2005)</h1>
    <p>{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}</p>
    <h2>GFDL-CM3 (historical, 1979-2005)</h1>
    <p>{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}</p>
    </body>
    </html>""".format(*img_tags)
    f.write(message)
    f.close()
    print('done.')

if __name__ == '__main__':
    import base64
    import numpy as np
    import os
    import pandas as pd
    import xarray as xr
    import seaborn as sns
    import time
    from matplotlib import pyplot as plt
    from pyproj import Proj, transform

    val_dir = '/workspace/Shared/Tech_Projects/DOT/project_data/validate_wrf_estimates'
    wrf_dir = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_interval_durations/'
    # plots for ERA-Interim based estimates
    plot_deltas(wrf_dir, val_dir, 'ERA-Interim')
    # plots for GCM-based estimates
    plot_deltas(wrf_dir, val_dir, 'NCAR-CCSM4')
    plot_deltas(wrf_dir, val_dir, 'GFDL-CM3')

    #write_md(val_dir)
    write_html(val_dir)