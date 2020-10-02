# download Atlas 14 data
# store path to the download directory in the A14_DOWNLOAD_DIR env var

import os, shutil
import wget
from datetime import datetime

# web page access: https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html
# target directory: https://hdsc.nws.noaa.gov/pub/hdsc/data/ak/

# generate target urls
url = "https://hdsc.nws.noaa.gov/pub/hdsc/data/ak/ak{}yr{}a{}_ams.zip"
intervals = ["2", "5", "10", "25", "50", "100", "200", "500", "1000"]
durations = [
    "60m",
    "02h",
    "03h",
    "06h",
    "12h",
    "24h",
    "48h",
    "03d",
    "04d",
    "07d",
    "10d",
    "20d",
    "30d",
    "45d",
    "60d",
]
ests = ["", "u", "l"]
files = [
    url.format(interval, duration, est)
    for interval in intervals
    for duration in durations
    for est in ests
]

# iterate through files, check for absence, download
out_dir = os.getenv("A14_DOWNLOAD_DIR")
if out_dir == None:
    exit("define download directory for files in A14_DOWNLOAD_DIR")

start = datetime.now()

for file in files:
    # NOAA changed filenames from 01h to 60m, 
    #   this is done so as to avoid changing multiple pipeline scripts
    fn = os.path.basename(file).replace("60m", "01h") 
    out_fp = os.path.join(out_dir, fn)
    if not os.path.exists(out_fp):
        print(file)
        wget.download(file, out=out_fp)

end = datetime.now()
diff = end - start
print("All files downloaded for " + str(diff.seconds) + "s")

# # NOTES:
# - metadata: ftp://hdsc.nws.noaa.gov/pub/hdsc/data/metadata/ak_metadata.xml
#   - MAP VALUES: Grid cell precipitation in inches*1000
#   - MORE INFORMATION ON THE GRIDS:
#       - 1000ths of inches; -9 is missing
#       - akXyrYY (X = return period, YY = duration; u = upper limit, l= lower limit) for partial duration series, _ams for annual maximum series

# # # #
# This GIS grid atlas contains precipitation frequency estimates for Alaska based on precipitation data collected between 1886-2011.
# This atlas is an updated version of Technical Paper 47, published in 1963, and Technical Paper 52, published in 1965.
# The grids provide information for durations from 5 minutes through 60 days, and for return periods of 1 year through 1000 years.
# All grids are in geographic coordinate system (NAD83 horizontal datum) and units are in 1000th of inches.
