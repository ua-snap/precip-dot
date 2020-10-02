# reproject extracted Atlas 14 data to epsg:3338

import os, glob, subprocess
import multiprocessing as mp


def nad83_to_3338(fp, bounds):
    print(fp)

    dirname, basename = os.path.split(fp)

    output_path = dirname.replace("extracted", "warped")
    try:
        if not os.path.exists(output_path):
            _ = os.makedirs(output_path)
    except:
        pass

    del_files = []
    out_fp = os.path.join(output_path, basename.replace(".asc", ".tif"))
    _ = subprocess.call(
        [
            "gdalwarp",
            "-q",
            "-overwrite",
            "-srcnodata",
            "-9",
            "-dstnodata",
            "-9999",
            "-s_srs",
            "NAD83",
            "-t_srs",
            "EPSG:3338",
            "-co",
            "COMPRESS=LZW",
            "-te",
        ]
        + bounds
        + [fp, out_fp]
    )

    return out_fp


def run(x):
    return nad83_to_3338(*x)


if __name__ == "__main__":
    # list the data
    down_dir = os.getenv("A14_DOWNLOAD_DIR")
    raw_dir = os.path.dirname(down_dir)
    extr_dir = os.path.join(raw_dir, "extracted")
    extr_fps = glob.glob(os.path.join(extr_dir, "*.asc"))

    # only need to reproject if not present in warped
    warp_dir = os.path.join(raw_dir, "warped")
    warp_fns = [os.path.basename(fn) for fn in glob.glob(os.path.join(warp_dir, "*"))]
    extr_fps = [
        fp
        for fp in extr_fps
        if os.path.basename(fp).replace(".asc", ".tif") not in warp_fns
    ]
    # make warped dir if not present
    if not os.path.isdir(warp_dir):
        os.mkdir(warp_dir)

    # hardwire the bounds of AK extent (3338)
    bounds = [
        "-2176652.08815013",
        "405257.4902562425",
        "1501904.8634676873",
        "2384357.9149669986",
    ]

    # args
    args = [(fp, bounds) for fp in extr_fps]

    # parallel run
    pool = mp.Pool(30)
    out = pool.map(run, args)
    pool.close()
    pool.join()
