# make a slurm script
def write_slurm_script( fn, out_path, ncpus, slurm_fn ):
    head = '#!/bin/sh\n' + \
        '#SBATCH --ntasks={}\n'.format(ncpus) + \
        '#SBATCH --nodes=1\n' + \
        '#SBATCH --ntasks-per-node={}\n'.format(ncpus) + \
        '#SBATCH --account=snap\n' + \
        '#SBATCH --mail-type=FAIL\n' + \
        '#SBATCH --mail-user=malindgren@alaska.edu\n' + \
        '#SBATCH -p viz\n'

    SCRIPTNAME = '/workspace/UA/malindgren/repos/precip-dot/pipeline/compute_return_intervals_with_confbounds.py'
    command = ['ipython', SCRIPTNAME, '--', '-f', fn, '-o', out_path, '-n', str(ncpus)]
    commandstring = ' '.join(command)

    # change the dir to the slurm output path
    slurm_path, basename = os.path.split( slurm_fn )
    os.chdir( slurm_path )

    # write the .slurm file
    with open( slurm_fn, 'w' ) as f:
        f.write( head + '\n' + commandstring + ';\n' )

    return slurm_fn


if __name__ == '__main__':
    import os, glob

    # # set-up pathing and list files
    ncpus=63
    base_path = '/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt'
    ams_path=os.path.join(base_path, 'annual_maximum_series')
    out_path=os.path.join(base_path, 'output_interval_durations')
    slurm_path=os.path.join('/workspace/UA/malindgren/repos/precip-dot/pipeline/slurm_run_scripts')
    
    # list the AMS files
    files=sorted(glob.glob(os.path.join(ams_path,'pcpt_*_ams.nc')))

    for fn in files:
        slurm_basename = 'run_'+os.path.basename(fn).replace('pcpt','run').replace('.nc','.slurm')
        slurm_fn = os.path.join(slurm_path, slurm_basename)

        # make sure the slurm dir exists and create it if not.
        dirname = os.path.dirname(slurm_fn)
        if not os.path.exists(dirname):
            _ = os.makedirs(dirname)

        # write sbatch
        write_slurm_script( fn, out_path, ncpus, slurm_fn )
