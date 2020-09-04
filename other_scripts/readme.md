# other_scripts

A collection of other scripts used at one point or another in the process of working on the project, and may likely be useful again. The scripts and their utility are described here.

- **tif2nc.py**: Generic script to convert a GeoTiff file to NetCDF

## atlas14_data_access_preprocessing

Scripts to download and extract NOAA Atlas 14 data from NOAA.

- **download_noaa_atlas14_data.py**: Download relevant Alaska data from NOAA's FTP server.
- **unzip_downloaded_atlas14_data.py**: Extract the downloaded zip files
- **project_grid_to_akalbers_noaa_atlas14_data.py**: Convert NOAA data to GeoTiff projected to **EPSG:3338**. The output from this is what the pipeline expects to find in the `NOAA` data directory.

## reporting

Scripts used for processing and summarizing elements of the output data.

- **ams_summary.py**: For a particular x/y coordinate, print a table of all of the AMS data at that point
- **get_average_ci_width.py**: Get the average width of the confidence intervals in a return intervals file.
- **percentiles.py**: Get the percentiles of the data (across the entire map) for a series of final output files.
- **point_data.py**: For a particular x/y coordinate, summarize all of the final data for that point.
- **point_data_general.py**: Like point_data, but can extract data for some of the other pipeline steps
- **point_data_noaa.py**: Like the other ones, but gets data from the original NOAA Atlas 14 data.
- **random-sample.pl**: Get a random sample of the data files across the entire pipeline representing all of the data used in a particular subset of the pipeline.
