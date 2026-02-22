# /// script
# dependencies = [
#   "pillow",
#   "numpy",
#   "opencv-python",
# ]
# ///


import argparse
from PIL import Image
import numpy as np
from time import time
from cv2 import bilateralFilter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply OpenCV bilateral filter to an image.")
    parser.add_argument("outfile", nargs="?", help="Output file path", default="tests/out/img_filtered_cv.png")
    parser.add_argument("infile", nargs="?", help="Input file path", default="tests/data/blobs.jpg")
    parser.add_argument("k", nargs="?", type=int, help="Diameter of each pixel neighborhood", default=21)
    parser.add_argument("sigmaColor", nargs="?", type=float, help="Filter sigma in color space", default=75)
    parser.add_argument("sigmaSpace", nargs="?", type=float, help="Filter sigma in coordinate space", default=75)
    parser.add_argument("-o", "--outfile", dest="outfile_opt", help="Output file path (override positional)")
    parser.add_argument("-i", "--infile", dest="infile_opt", help="Input file path (override positional)")
    parser.add_argument("-k", dest="k_opt", type=int, help="Diameter of each pixel neighborhood (override positional)")
    parser.add_argument(
        "--sigmaColor", dest="sigmaColor_opt", type=float, help="Filter sigma in color space (override positional)"
    )
    parser.add_argument(
        "--sigmaSpace", dest="sigmaSpace_opt", type=float, help="Filter sigma in coordinate space (override positional)"
    )
    parser.add_argument("--bench", nargs="?", const=100, type=int, help="Run benchmark for N runs (default 100)")
    args = parser.parse_args()

    # Use optional args if provided
    outfile = args.outfile_opt if args.outfile_opt else None
    infile = args.infile_opt if args.infile_opt else args.infile
    k = args.k_opt if args.k_opt is not None else args.k
    sigmaColor = args.sigmaColor_opt if args.sigmaColor_opt is not None else args.sigmaColor
    sigmaSpace = args.sigmaSpace_opt if args.sigmaSpace_opt is not None else args.sigmaSpace

    bench_N = args.bench if args.bench is not None else 1

    # Load the image
    img = Image.open(infile).convert("RGB")
    img_np = np.array(img)

    start_time = time()
    filtered_img_cv = bilateralFilter(img_np, k, sigmaColor, sigmaSpace)
    for i in range(bench_N - 1):
        filtered_img_cv = bilateralFilter(img_np, k, sigmaColor, sigmaSpace)
    end_time = time()

    if outfile is not None:
        out = Image.fromarray(filtered_img_cv)
        out.save(outfile)

    if bench_N > 1:
        print(f"{img_np.shape}, k: {k}, sigma_range: {sigmaColor}, sigma_spatial: {sigmaSpace}")
        print(f"CV BF: {(end_time - start_time) / bench_N:.6f}s")
