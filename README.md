# OpenCL vs CPU vs Numba grayscale benchmark

This repository converts `gigapixel.jpg` to grayscale using four approaches:

- A pure Python double loop over the 2D image.
- A flattened 1D Python loop.
- An OpenCL kernel (requires a working OpenCL device and `pyopencl`).
- A Numba-accelerated double loop.

## Requirements

- Python 3.12+
- `numpy`, `numba`, `imageio`
- Optional (only needed if you want to run the original script end-to-end): `pyopencl`, `opencv-python` (or `opencv-python-headless` for headless environments), `matplotlib`

Install the essentials with:

```bash
pip install numpy numba imageio
```

To try the GPU path and the OpenCV previews, also install:

```bash
pip install pyopencl opencv-python-headless matplotlib
```

## How to run

The provided `benchmark.py` script expects a display for `cv2.imshow` and an OpenCL device for the GPU section:

```bash
python benchmark.py
```

If you are on a headless machine or do not have an OpenCL device, you can still exercise the CPU and Numba paths by running just the computational sections (e.g., by commenting out the `cv2.imshow` calls). The measurements below were collected that way while keeping the input image unchanged.

## Results from a recent run

Measured on a GitHub-hosted runner (AMD EPYC 7763, 4 vCPUs, Python 3.12) using the full `gigapixel.jpg` image. Times are wall-clock seconds.

| Method                  | Time (s) | Notes                                          |
| ----------------------- | -------- | ---------------------------------------------- |
| CPU nested 2D loops     | 35.97    | Pure Python double loop over height and width. |
| CPU flattened 1D loop   | 25.35    | Loops over flattened arrays.                   |
| Numba @njit 2D loop     | 0.33     | First call includes JIT compilation overhead.  |
| OpenCL kernel           | n/a      | Not executed here (no OpenCL device available).|

Even with the large image, Numba delivered two orders of magnitude faster execution than pure Python loops. The OpenCL path could not be exercised on the available runner; if you have an OpenCL-capable GPU or CPU runtime, install `pyopencl` and rerun `benchmark.py` to compare its performance.
