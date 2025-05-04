# OpenCV incremental `cv::remap()` benchmark

Demonstrates that [`cv::remap()`](https://docs.opencv.org/4.11.0/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4) can be called incrementally over the rows of an image.

This saves memory, not keeping the whole _map_ in memory.

* Example: If your `dst` image is 16Ki * 8Ki pixels (128 MiP) RGB, that's 384 MiB in memory.
* The corresponding _map_ is 128 MiP * `sizeof(CV_32FC2)` = 1024 MiB.
  * You can reduce this by e.g. **16x** with small performance impact by remapping groups of 512 rows.

The outputs are identical to full `remap()` (the code checks this).

This benchmark shows the performance impact.


## Building

Install `opencv4`, then run:

```sh
g++ -O2 -g -Wall -std=c++17 opencv-row-by-row-remap.cpp $(pkg-config --cflags --libs opencv4 eigen3) -o opencv-row-by-row-remap && ./opencv-row-by-row-remap
```


## Example output

On `AMD Ryzen 7 7700X` (8 real cores, 16 threads):

```
dest_rows: 8000
dest_cols: 10000

Interpolation: INTER_LINEAR
Full remap took            32 ms
Single-row-incremental remap took 302 ms
multi_row_step: 128
Multi-row-incremental (range-based) remap took 50 ms
Multi-row-incremental (ROI based) remap took 50 ms
Multi-row-incremental (range-based, copying) remap took 119 ms
multi_row_step: 256
Multi-row-incremental (range-based) remap took 44 ms
Multi-row-incremental (ROI based) remap took 43 ms
Multi-row-incremental (range-based, copying) remap took 105 ms
multi_row_step: 512
Multi-row-incremental (range-based) remap took 39 ms
Multi-row-incremental (ROI based) remap took 38 ms
Multi-row-incremental (range-based, copying) remap took 110 ms

Interpolation: INTER_LANCZOS4
Full remap took            555 ms
Single-row-incremental remap took 4446 ms
multi_row_step: 128
Multi-row-incremental (range-based) remap took 701 ms
Multi-row-incremental (ROI based) remap took 695 ms
Multi-row-incremental (range-based, copying) remap took 776 ms
multi_row_step: 256
Multi-row-incremental (range-based) remap took 620 ms
Multi-row-incremental (ROI based) remap took 616 ms
Multi-row-incremental (range-based, copying) remap took 695 ms
multi_row_step: 512
Multi-row-incremental (range-based) remap took 574 ms
Multi-row-incremental (ROI based) remap took 580 ms
Multi-row-incremental (range-based, copying) remap took 650 ms
./opencv-row-by-row-remap  92.31s user 5.30s system 641% cpu 15.219 total
```


## Remarks

* The benchmark code actually keeps the whole _map_ in memory (`cv::Mat_<cv::Vec<float, 2>> mapping`).
  This is for code simplicity.
  The benchmark loops access the `mapping` incrementally, so it could easily be computed on the fly.


## Analysis

* `remap()` is implemented using multi-core speedups.
* `Single-row-incremental` is very slow, because it apparently does not multi-core.
* `Multi-row-incremental` works well for larger row blocks of e.g. 512 with only
  * ~21% overhead for `LINEAR` interpolation
  * ~3% overhead for `LANCZOS4` interpolation
* The slower the interpolation function is, the lower performance impact of incremental remapping.
