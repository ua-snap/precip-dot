# How To Install and Use Python on Atlas

## login to the atlas head node
your username / password are from UA Credentials -- See ELMO
```sh
ssh atlas.snap.uaf.edu
```

## download and install python3
### download python
```sh
mkdir ~/src
mkdir ~/.localpython
cd ~/src
wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tar.xz
```

### unzip 
```sh
tar xvfJ Python-3.7.4.tar.xz
```

### python3 installation / configuration
```sh
cd ~/src/Python-3.7.4
make clean
./configure --prefix=/home/UA/malindgren/.localpython
make
make install
```

## use the installation
### make a virtual environment where we will install some packages
```sh
~/.localpython/bin/python3.7 -m venv ~/venv
source ~/venv/bin/activate
```

### install some packages 
```sh
pip install --upgrade pip
pip install numpy
pip install 'ipython[all]'
pip install scipy rasterio fiona pandas geopandas scikit-image scikit-learn shapely netCDF4 xarray
```

