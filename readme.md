# Precip-DOT

These data constitute an effort to update the NOAA Atlas 14 for Alaska using more recent historical observed climate information and adding in the effects of a changing climate. The goal is to produce a similar set of data as is displayed and served currently through the [NOAA Atlas 14 Precipitation Frequency Web Interface](https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_map_ak.html), using the WRF Dynamically Downscaled Data for Alaska produced by researchers affiliated with the Alaska Climate Adaptation Science Center (AK-CASC). This work is funded by the Alaska Department of Transportation (AKDOT) to add climate futures data to existing workflows designing culverts and other engineering features of road construction.

# The Data Pipeline

The data transformation in this project follows a pipeline with the following steps: 

1. **Starting point:** WRF hourly pcpt data (can be obtained [here](http://wrf-ak-ar5.s3-website-us-east-1.amazonaws.com/))
2. **Durations:** Calculate duration series for various time periods.
3. **AMS:** Calculate **A**nnual **M**aximum **S**eries for all duration series.
4. **Intervals:** Calculate return intervals based off AMS.

The WRF data includes 5 different models/data groups (listed below). The pipeline is repeated for every group:

* NCAR-CCSM4_historical
* NCAR-CCSM4_rcp85
* ERA-Interim_historical
* GFDL-CM3_historical
* GFDL-CM3_rcp85

The python scripts in the [pipeline directory](pipeline/) are used to execute different steps of the pipeline and the [run-pipeline](run-pipeline) script is used to execute all or part of the entire pipeline.

# Running the Pipeline

## Setting Up

If you're running on SNAP's ATLAS server, then some of this has probably already been done for you.

1. **Install Dependencies:** Call `pip install -r requirements.txt` to install all of the necessary python packages (it is recommended that you do this inside a new virtual environment or conda environment). You may also need to install various libraries such as HDF5 and NetCDF to get the packages to install.
2. **Set up Data Directory:** Decide on the directory where you want to store all the data. In that directory, create another directory called `pcpt` and download all of hourly pcpt WRF data into it. (The data can be found on S3 [here](http://wrf-ak-ar5.s3-website-us-east-1.amazonaws.com/)).

## Executing the Pipeline

Simply calling `./run-pipeline` will execute the entire pipeline for all data sets with default options configured for running on SNAP's ATLAS server (such as using the SLURM cluster and using `/workspace/Shared/Tech_Projects/DOT/project_data/wrf_pcpt/` as the data directory). However, the [run-pipeline](run-pipeline) script is very powerful and through various command-line options, it can be customized to run just subsets of the pipeline, in different directories, with or without SLURM and more. Call `./run-pipeline --help` for more information.

# Also in this repository

* **ancillary:** ???????
* **documents:** Scraps of project documentation.
* **eva_tutorials_web:** Completed tutorials for extreme value analysis code and other practice scripts.
* **other_scripts:** Other data processing scripts that were used at one point or another.
* **R_workspace:** Example data and R scripts.
