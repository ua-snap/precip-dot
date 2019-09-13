from ftplib import FTP
from datetime import datetime
import pandas as pd
import os

# set up the FTP stuff -- see: https://hdsc.nws.noaa.gov/hdsc/pfds/pfds_gis.html
start = datetime.now()
ftp = FTP('hdsc.nws.noaa.gov')
ftp.login('anonymous','malindgren@alaska.edu')
ftp.cwd('pub/hdsc/data/ak')

# Get All Files filtered to the .zip's
files = [ fn for fn in ftp.nlst() if '.zip' in fn ]

# filter to the ones we actually want
intervals = ['60m','2h','3h','6h','12h','24h','2d','3d','4d','7d','10d','20d','30d','45d','60d',]
files = [ fn for i in intervals for fn in files if i in fn ]

os.chdir( '/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/raw/zip' )

for file in files:
	if not os.path.exists(file):
		print("Downloading..." + file)
		ftp.retrbinary("RETR " + file , open(file, 'wb').write)

ftp.close()

end = datetime.now()
diff = end - start
print('All files downloaded for ' + str(diff.seconds) + 's')


# # NOTES:
# - metadata: ftp://hdsc.nws.noaa.gov/pub/hdsc/data/metadata/ak_metadata.xml
# 	- MAP VALUES: Grid cell precipitation in inches*1000
# 	- MORE INFORMATION ON THE GRIDS:
# 		- 1000ths of inches; -9 is missing
# 		- akXyrYY (X = return period, YY = duration; u = upper limit, l= lower limit) for partial duration series, _ams for annual maximum series

# # # # 
# This GIS grid atlas contains precipitation frequency estimates for Alaska based on precipitation data collected between 1886-2011.
# This atlas is an updated version of Technical Paper 47, published in 1963, and Technical Paper 52, published in 1965. 
# The grids provide information for durations from 5 minutes through 60 days, and for return periods of 1 year through 1000 years. 
# All grids are in geographic coordinate system (NAD83 horizontal datum) and units are in 1000th of inches.


