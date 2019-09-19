# Descriptions of processing `pipeline` files:

- `compute_return_intervals_with_confbounds.py`
this script runs the return intervals computation on the annual maximum series for a given duration. Including an experimental (read:not-properly-working) method(s) for computing the confidence bounds using a bootstrapping approach.
- `compute_return_intervals_with_confbounds_SEPT2019.py`
- `make_durations_series_wrf_data.py`
compute the durations series in a cascading way from the shortest duration to longest.
- `make_annual_maximum_series_wrf_data.py`
compute the annual maximum series from the pre-computed durations series, which are used in l-moments and curve fitting estimation.
- `make_slurm_run_scripts_return_intervals.py`
this script will generate a directory structure with a bunch of SLURM sbatch files that are used to perform the above operations on a given set of data.
- `run_all_slurmscripts.py`
a simple script that launches all of the scripts into the SLURM scheduler on Atlas that are produced in the `make_slurm_run_scripts_return_intervals.py` script.
- `run_pipeline_dot_precip.py`
this is the script that runs all of the others and will run the computation from the "raw" WRF Precipitation (PCPT) data to the outputs required for this project. This leverages the Atlas cluster to do the work.
- `compute_lmoments_wrf_pcpt.py`
return l-moment statistics in the order: [l1, l2, t3, t4, t5] for a given duration/annual maximum series combination and dump out to multiband GeoTiffs. This is useful for comparison of the lmoments statistics with those that are distributed with the NOAA Atlas 14 Documents.  They are in an addendum in the back of the main report.

