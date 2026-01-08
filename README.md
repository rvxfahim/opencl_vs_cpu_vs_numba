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

If you are on a headless machine or do not have an OpenCL device, you can still exercise the CPU and Numba paths by running just the computational sections (e.g., by commenting out the `cv2.imshow` calls). Feel free to record your own timings on `gigapixel.jpg` (5627x10000, uint8 RGB, ~2.4 MB) or any other image to compare approaches on your hardware.
