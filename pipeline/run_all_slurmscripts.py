import os, glob

os.chdir('/workspace/UA/malindgren/repos/precip-dot/pipeline/slurm_run_scripts')

for fn in glob.glob('./*.slurm'):
	os.system('sbatch {}'.format(fn))
