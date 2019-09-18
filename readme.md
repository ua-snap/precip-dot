# Precip-DOT

## To Use This Package:

### Install Python (if you dont have it already)
For installation procedures on the SNAP Atlas Cluster see [here](https://github.com/ua-snap/precip-dot/blob/master/other_scripts/How_To_Install_and_Use_Python_on_Atlas.md)

### Clone the Repository from github (this assumes you have a GitHub account and Git CLI installed)
```sh
git clone git@github.com:ua-snap/precip-dot.git
cd precip-dot
```

### Generate a Virtual Environment with this new (or existing) Python3 installation
```sh
# this follows the pathing and location of python3 following the above install steps
~/.localpython/bin/python3.7 -m venv venv

# source in the new venv
source venv/bin/activate

# install (using `pip`) the needed packages
pip install -r requirements.txt
```

### Run the Pipeline
There is a launcher script that will run all of the processing for this project.  It is located [here](https://github.com/ua-snap/precip-dot/blob/master/pipeline/run_pipeline_dot_precip.py) or in the `pipeline` sub-folder in the `precip-dot` repository you cloned above. You might have to fiddle around with the pathing therein as it is setup to work on a specific system with specific named servers and folder paths.

I would instantiate the processing using the following:

```sh
# login to atlas
ssh atlas.snap.uaf.edu

# go to the repository
cd /path/to/repo/precip-dot

# activate the virtual environment
source venv/bin/activate

# now go to the pipeline
cd pipeline

# example 1: run with sbatch
sbatch -p 'main' ipython run_pipeline_dot_precip.py

# example 2: run on a single node (until other process launch on other nodes)
screen srun -p main -N 1 -n 32 -w atlas01 --pty /bin/bash
ipython run_pipeline_dot_precip.py

``` 

# Diagram of Processing Workflow As It Currently Stands:

![processflow](https://github.com/ua-snap/precip-dot/blob/master/documents/DOT_Project_ProcessFlow.png)
