# unzip the downloaded data (in parallel)
# currently setup to unzip files not present in "warped" directory
# use A14_DOWNLOAD_DIR env var (see download_noaa_atlas14_data.py)

import os, glob
import multiprocessing as mp


def unzip(args):
    fp, extr_dir = args[0], args[1]
    print("Extracting:", fp)
    command = "unzip {} -d {}".format(fp, extr_dir)
    return os.system(command)


if __name__ == "__main__":
    # list the data
    down_dir = os.getenv("A14_DOWNLOAD_DIR")
    raw_dir = os.path.dirname(down_dir)
    down_fps = glob.glob(os.path.join(raw_dir, "zip", "*.zip"))

    # only need to extract if not present in warped
    warp_dir = os.path.join(raw_dir, "warped")
    warp_fns = [os.path.basename(fn) for fn in glob.glob(os.path.join(warp_dir, "*"))]
    down_fps = [
        fp
        for fp in down_fps
        if os.path.basename(fp).replace(".zip", ".tif") not in warp_fns
    ]
    
    extr_dir = os.path.join(raw_dir, "extracted")
    if not os.path.isdir(extr_dir):
        os.mkdir(extr_dir)
    
    # pool args of each zip filepath (unique) and out dir
    args = [(fp, extr_dir) for fp in down_fps]
    pool = mp.Pool(32)
    pool.map(unzip, args)
    out = pool.close()
    pool.join()
