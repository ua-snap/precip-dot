# # # # # # # # # # # # # # # # # # # # # # # # # 
# # NOAA Atlas 14 data -- reproject to EPSG:3338
# # # # # # # # # # # # # # # # # # # # # # # # #

def nad83_to_3338( fn, bounds ):
    print(fn)

    dirname, basename = os.path.split(fn)

    output_path = dirname.replace('extracted', 'warped')
    try:
        if not os.path.exists(output_path):
            _ = os.makedirs(output_path)
    except:
        pass

    del_files = []
    out_fn = os.path.join(output_path, basename.replace('.asc', '.tif'))
    _ = subprocess.call(['gdalwarp', '-q', '-overwrite', '-srcnodata', '-9','-dstnodata', '-9999', '-s_srs', 'NAD83', '-t_srs', 'EPSG:3338', '-co', 'COMPRESS=LZW', '-te'] + bounds + [fn, out_fn ])

    return out_fn

def run( x ):
    return nad83_to_3338(*x)

if __name__ == '__main__':
    import os, glob, subprocess, fiona, copy
    import multiprocessing as mp

    path = '/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/raw/extracted'
    shp_fn = '/workspace/Shared/Tech_Projects/DOT/project_data/NOAA_Atlas14/shapefiles/AK_Extent_3338.shp'
    
    # with fiona.open( shp_fn ) as shp:
    #     bounds = [str(i) for i in shp.bounds] # to pass to gdalwarp

    # hardwire the bounds since I buffered the above by 10k meters on each side to make sure we get it all.
    bounds = ['-2176652.08815013', '405257.4902562425', '1501904.8634676873', '2384357.9149669986']

    # list the files:
    files = glob.glob(os.path.join(path, '*.asc'))

    # args
    args = [ (fn,bounds) for fn in files ]

    # parallel run
    pool = mp.Pool(30)
    out = pool.map( run, args )
    pool.close()
    pool.join()

