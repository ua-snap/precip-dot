# # DOT Precipitation Frequencies from WRF PCPT data
# #  --> run processing pipeline for all data_groups
# # Author: Michael Lindgren (malindgren@alaska.edu)

if __name__ == '__main__':
    import os, glob, subprocess

    # change the directory to that where the code is stored.
    os.chdir('/workspace/UA/malindgren/repos/precip-dot/pipeline')
    data_groups = [ 'NCAR-CCSM4_historical',
                    'NCAR-CCSM4_rcp85',
                    'ERA-Interim_historical',
                    'GFDL-CM3_historical',
                    'GFDL-CM3_rcp85' ]
    
    ncpus = 63
    for data_group in data_groups:
        print('running: {}'.format(data_group))

        # generate duration series
        print('computing duration series')
        path = '/rcs/project_data/wrf_data/hourly_fix/pcpt'
        durations_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/durations'
        _ = subprocess.call(['ipython','make_durations_series_wrf_data.py','--', \
                                '-p', path, '-o', durations_path, '-d', data_group ])
        # generate annual maximum series from durations
        print('computing ams series')
        ams_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/annual_maximum_series'
        _ = subprocess.call(['ipython','make_annual_maximum_series_wrf_data.py','--',\
                                '-p', durations_path, '-o', ams_path, '-d', data_group])

        # compute lmoment-based return interval predictions (with confidence bounds) -- launch with slurm
        print('computing return intervals and confidence bounds')
        output_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/output_interval_durations'
        files = glob.glob(os.path.join(ams_path , '*{}*.nc'.format(data_group)))
        for fn in files:
            # launch using SLURM's sbatch command (will queue jobs on the 'main' partition using one node (-N 1))
            _ = subprocess.call(['sbatch','-p','main,viz','-N','1','ipython','compute_return_intervals_with_confbounds.py', '--', \
                                '-fn', fn, '-o', output_path])
        
