import time
import cv2
import pyopencl
import numpy as np
import imread
import matplotlib.pyplot as plt
from numba import njit

# read image using imread
img = imread.imread('gigapixel.jpg')
# get red channel
r = np.array(img[:, :, 0], dtype=np.float32)
# get green channel
g = np.array(img[:, :, 1], dtype=np.float32)
# get blue channel
b = np.array(img[:, :, 2], dtype=np.float32)
# define gray channel
gray = np.empty_like(r)
# print shape of r g b
print(r.shape)
print(g.shape)
print(b.shape)
# print shape of gray
print(gray.shape)
# without gpu
start = time.time()
for i in range(r.shape[0]):
    for j in range(r.shape[1]):
        gray[i, j] = (r[i, j] + g[i, j] + b[i, j]) / 3
plt.imshow(gray)
plt.show()
# convert to uint8
gray = np.uint8(gray)
# save image
imread.imsave('gray_cpu.jpg', gray)
print('done using CPU')
print('time: ', time.time() - start)
# convert 2d array to 1d array of r g b
r = r.flatten()
g = g.flatten()
b = b.flatten()
# create empty array for gray
gray = np.empty_like(r)
# convert to gray using cpu
start = time.time()
for i in range(r.shape[0]):
    gray[i] = (r[i] + g[i] + b[i]) / 3
# convert to uint8
gray = np.uint8(gray)
# reshape gray to 2d array
gray = gray.reshape(img.shape[0], img.shape[1])
# save image
imread.imsave('gray_cpu_reshaped.jpg', gray)
print('done flattened method using CPU')
print('time taken: ', time.time() - start)
gray = np.empty_like(r)
# convert to gray using gpu
start = time.time()
ocl_platforms = (platform.name for platform in pyopencl.get_platforms())
print("\n".join(ocl_platforms))
# select platform
platform = pyopencl.get_platforms()[0]
# select device
device = platform.get_devices()[0]

# create context
ctx = pyopencl.Context(devices=[device])
# create queue
queue = pyopencl.CommandQueue(ctx)
# create buffer
r_buf = pyopencl.Buffer(ctx, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR, hostbuf=r)
g_buf = pyopencl.Buffer(ctx, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR, hostbuf=g)
b_buf = pyopencl.Buffer(ctx, pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR, hostbuf=b)
gray_buf = pyopencl.Buffer(ctx, pyopencl.mem_flags.WRITE_ONLY, gray.nbytes)
# create program
program = pyopencl.Program(ctx, """
    __kernel void rgb2gray(__global float *r, __global float *g, __global float *b, __global float *gray) {
        int i = get_global_id(0);
        gray[i] = (r[i] + g[i] + b[i]) / 3;
    }
    """).build()
# execute kernel
program.rgb2gray(queue, gray.shape, None, r_buf, g_buf, b_buf, gray_buf)
# copy result from buffer to host
pyopencl.enqueue_copy(queue, gray, gray_buf)
# reshape gray to 2d array
gray = gray.reshape(img.shape[0], img.shape[1])
# to uint8
gray = np.uint8(gray)
# save image
imread.imsave('gray_gpu.jpg', gray)
print('done using GPU')
print('time taken: ', time.time() - start)

# get red channel
r = np.array(img[:, :, 0], dtype=np.float32)
# get green channel
g = np.array(img[:, :, 1], dtype=np.float32)
# get blue channel
b = np.array(img[:, :, 2], dtype=np.float32)
# define gray channel
gray = np.empty_like(r)

# using numba

@njit
def rgb2gray(r, g, b, gray_):
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            gray_[i, j] = (r[i, j] + g[i, j] + b[i, j]) / 3
    return gray_


# execute rgb2gray
start = time.time()
gray = rgb2gray(r, g, b, gray)
# convert to uint8
gray = np.uint8(gray)
# save image
imread.imsave('gray_numba.jpg', gray)
print('done using numba')
print('time taken: ', time.time() - start)
#show output images in 4 seperate windows of size 512x512 using opencv
cv2.namedWindow('CPU', cv2.WINDOW_NORMAL)
cv2.namedWindow('CPU_flattened', cv2.WINDOW_NORMAL)
cv2.namedWindow('GPU', cv2.WINDOW_NORMAL)
cv2.namedWindow('Numba', cv2.WINDOW_NORMAL)
cv2.resizeWindow('CPU', 512, 512)
cv2.resizeWindow('CPU_flattened', 512, 512)
cv2.resizeWindow('GPU', 512, 512)
cv2.resizeWindow('Numba', 512, 512)
cv2.imshow('CPU', imread.imread('gray_cpu.jpg'))
cv2.imshow('CPU_flattened', imread.imread('gray_cpu_reshaped.jpg'))
cv2.imshow('GPU', imread.imread('gray_gpu.jpg'))
cv2.imshow('Numba', imread.imread('gray_numba.jpg'))
cv2.waitKey(0)
cv2.destroyAllWindows()
#end