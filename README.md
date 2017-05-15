# CV_examples
This is my CV examples

## Python OpenCV
[python opencv](./PythonOpenCV/README.md)

## Yum On Centos
sudo yum -y update  
yum list availabel opencv\*   
sudo yum install opencv opencv-core opencv-devel opencv-devel-docs opencv-python  

## Compile OpenCv-3.2 to Python3 On CentOS7
### Prereqs
sudo yum-builddep -y opencv-devel  
sudo yum install cmake git pkgconfig libpng-devel libjpeg-turbo-devel jasper-devel openexr-devel libtiff-devel libwebp-devel libdc1394-devel libv4l-devel gstreamer-plugins-base-devel gtk2-devel tbb-devel eigen3-devel   
sudo pip install numpy  

### Compile Opencv-3.2
git clone https://github.com/opencv/opencv.git  
git clone https://github.com/opencv/opencv_contrib.git  
cd opencv_contrib  
git checkout 3.2.0  
cd ../opencv  
git checkout 3.2.0  
mkdir release  
cd release  
cmake .. //cmake options  
make -j$(($(nproc) + 1))  
sudo make install   
sudo ldconfig  

## Compile Opencv-3.2 to Python3 on Ubuntu 16.04
### Prereqs
sudo apt-get install --assume-yes build-essential cmake git
sudo apt-get install --assume-yes build-essential pkg-config unzip ffmpeg qtbase5-dev
sudo apt-get install --assume-yes libopencv-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev
sudo apt-get install --assume-yes libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt-get install --assume-yes libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev
sudo apt-get install --assume-yes libvorbis-dev libxvidcore-dev v4l-utils
sudo pip3 install numpy

## cmake options

```
// Path to a program.
ANT_EXECUTABLE:FILEPATH=ANT_EXECUTABLE-NOTFOUND

// Build CUDA modules stubs when no CUDA SDK
BUILD_CUDA_STUBS:BOOL=OFF

// Create build rules for OpenCV Documentation
BUILD_DOCS:BOOL=ON

// Build all examples
BUILD_EXAMPLES:BOOL=OFF

// Build libjasper from source
BUILD_JASPER:BOOL=OFF

// Build libjpeg from source
BUILD_JPEG:BOOL=OFF

// Build openexr from source
BUILD_OPENEXR:BOOL=OFF

// Enables 'make package_source' command
BUILD_PACKAGE:BOOL=ON

// Build performance tests
BUILD_PERF_TESTS:BOOL=ON

// Build libpng from source
BUILD_PNG:BOOL=OFF

// Build shared libraries (.dll/.so) instead of static ones (.lib/.a)
BUILD_SHARED_LIBS:BOOL=ON

// Download and build TBB from source
BUILD_TBB:BOOL=OFF

// Build accuracy & regression tests
BUILD_TESTS:BOOL=ON

// Build libtiff from source
BUILD_TIFF:BOOL=OFF

// Include debug info into debug libs (not MSCV only)
BUILD_WITH_DEBUG_INFO:BOOL=ON

// Enables dynamic linking of IPP (only for standalone IPP)
BUILD_WITH_DYNAMIC_IPP:BOOL=OFF

// Build zlib from source
BUILD_ZLIB:BOOL=OFF

// Build utility applications (used for example to train classifiers)
BUILD_opencv_apps:BOOL=ON

// Include opencv_calib3d module into the OpenCV build
BUILD_opencv_calib3d:BOOL=ON

// Include opencv_core module into the OpenCV build
BUILD_opencv_core:BOOL=ON

// Include opencv_features2d module into the OpenCV build
BUILD_opencv_features2d:BOOL=ON

// Include opencv_flann module into the OpenCV build
BUILD_opencv_flann:BOOL=ON

// Include opencv_highgui module into the OpenCV build
BUILD_opencv_highgui:BOOL=ON

// Include opencv_imgcodecs module into the OpenCV build
BUILD_opencv_imgcodecs:BOOL=ON

// Include opencv_imgproc module into the OpenCV build
BUILD_opencv_imgproc:BOOL=ON

// Include opencv_ml module into the OpenCV build
BUILD_opencv_ml:BOOL=ON

// Include opencv_objdetect module into the OpenCV build
BUILD_opencv_objdetect:BOOL=ON

// Include opencv_photo module into the OpenCV build
BUILD_opencv_photo:BOOL=ON

// Include opencv_python2 module into the OpenCV build
BUILD_opencv_python2:BOOL=ON

// Include opencv_python3 module into the OpenCV build
BUILD_opencv_python3:BOOL=ON

// Include opencv_shape module into the OpenCV build
BUILD_opencv_shape:BOOL=ON

// Include opencv_stitching module into the OpenCV build
BUILD_opencv_stitching:BOOL=ON

// Include opencv_superres module into the OpenCV build
BUILD_opencv_superres:BOOL=ON

// Include opencv_ts module into the OpenCV build
BUILD_opencv_ts:BOOL=ON

// Include opencv_video module into the OpenCV build
BUILD_opencv_video:BOOL=ON

// Include opencv_videoio module into the OpenCV build
BUILD_opencv_videoio:BOOL=ON

// Include opencv_videostab module into the OpenCV build
BUILD_opencv_videostab:BOOL=ON

// Include opencv_world module into the OpenCV build
BUILD_opencv_world:BOOL=OFF

// Path to a program.
CCACHE_PROGRAM:FILEPATH=CCACHE_PROGRAM-NOTFOUND

// clAmdFft include directory
CLAMDBLAS_INCLUDE_DIR:PATH=CLAMDBLAS_INCLUDE_DIR-NOTFOUND

// AMD FFT root directory
CLAMDBLAS_ROOT_DIR:PATH=CLAMDBLAS_ROOT_DIR-NOTFOUND

// clAmdFft include directory
CLAMDFFT_INCLUDE_DIR:PATH=CLAMDFFT_INCLUDE_DIR-NOTFOUND

// AMD FFT root directory
CLAMDFFT_ROOT_DIR:PATH=CLAMDFFT_ROOT_DIR-NOTFOUND

// Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.
CMAKE_BUILD_TYPE:STRING=

// Configs
CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release

// Installation Directory
CMAKE_INSTALL_PREFIX:PATH=/usr/local

// Generate and parse .cubin files in Device mode.
CUDA_BUILD_CUBIN:BOOL=OFF

// Build in Emulation mode
CUDA_BUILD_EMULATION:BOOL=OFF

// Host side compiler used by NVCC
CUDA_HOST_COMPILER:FILEPATH=/usr/bin/gcc

// Path to a file.
CUDA_SDK_ROOT_DIR:PATH=CUDA_SDK_ROOT_DIR-NOTFOUND

// Compile CUDA objects with separable compilation enabled.  Requires CUDA 5.0+
CUDA_SEPARABLE_COMPILATION:BOOL=OFF

// Toolkit location.
CUDA_TOOLKIT_ROOT_DIR:PATH=CUDA_TOOLKIT_ROOT_DIR-NOTFOUND

// Print out the commands run while compiling the CUDA source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can be forced on with this option.
CUDA_VERBOSE_BUILD:BOOL=OFF

// Download external test data (Python executable and OPENCV_TEST_DATA_PATH environment variable may be required)
DOWNLOAD_EXTERNAL_TEST_DATA:BOOL=OFF

// The path to Eigen3/Eigen2 headers
EIGEN_INCLUDE_PATH:PATH=/usr/include/eigen3

// Enable AVX instructions
ENABLE_AVX:BOOL=OFF

// Enable AVX2 instructions
ENABLE_AVX2:BOOL=OFF

// Use ccache
ENABLE_CCACHE:BOOL=ON

// Enable coverage collection with  GCov
ENABLE_COVERAGE:BOOL=OFF

// Enable -ffast-math (not recommended for GCC 4.6.x)
ENABLE_FAST_MATH:BOOL=OFF

// Enable FMA3 instructions
ENABLE_FMA3:BOOL=OFF

// Collect implementation data on function call
ENABLE_IMPL_COLLECTION:BOOL=OFF

// Instrument functions to collect calls trace and performance
ENABLE_INSTRUMENTATION:BOOL=OFF

// Show all warnings even if they are too noisy
ENABLE_NOISY_WARNINGS:BOOL=OFF

// Enable -fomit-frame-pointer for GCC
ENABLE_OMIT_FRAME_POINTER:BOOL=ON

// Enable POPCNT instructions
ENABLE_POPCNT:BOOL=OFF

// Use precompiled headers
ENABLE_PRECOMPILED_HEADERS:BOOL=ON

// Enable profiling in the GCC compiler (Add flags: -g -pg)
ENABLE_PROFILING:BOOL=OFF

// Solution folder in Visual Studio or in other IDEs
ENABLE_SOLUTION_FOLDERS:BOOL=OFF

// Enable SSE instructions
ENABLE_SSE:BOOL=ON

// Enable SSE2 instructions
ENABLE_SSE2:BOOL=ON

// Enable SSE3 instructions
ENABLE_SSE3:BOOL=ON

// Enable SSE4.1 instructions
ENABLE_SSE41:BOOL=OFF

// Enable SSE4.2 instructions
ENABLE_SSE42:BOOL=OFF

// Enable SSSE3 instructions
ENABLE_SSSE3:BOOL=OFF

// Output directory for applications
EXECUTABLE_OUTPUT_PATH:PATH=/opt/opencv-3.2.0/release/bin

// Generate XML file for abi_compliance_checker tool
GENERATE_ABI_DESCRIPTOR:BOOL=OFF

// True if MKL found
HAVE_MKL:BOOL=OFF

// Change install rules to build the distribution package
INSTALL_CREATE_DISTRIB:BOOL=OFF

// Install C examples
INSTALL_C_EXAMPLES:BOOL=OFF

// Install Python examples
INSTALL_PYTHON_EXAMPLES:BOOL=OFF

// Install accuracy and performance test binaries and test data
INSTALL_TESTS:BOOL=OFF

// Enables mangled install paths, that help with side by side installs.
INSTALL_TO_MANGLED_PATHS:BOOL=OFF

// Alternative name of cblas.h
LAPACK_CBLAS_H:STRING=cblas.h

// Lapack implementation id
LAPACK_IMPL:STRING=OpenBLAS

// Path to BLAS include dir
LAPACK_INCLUDE_DIR:PATH=/usr/include

// Alternative name of lapacke.h
LAPACK_LAPACKE_H:STRING=lapacke.h

// Names of BLAS & LAPACK binaries (.so, .dll, .a, .lib)
LAPACK_LIBRARIES:STRING=/lib64/libopenblas.so

// Path to a file.
MKL_ROOT_DIR:PATH=MKL_ROOT_DIR-NOTFOUND

// Use MKL with OpenMP multithreading
MKL_WITH_OPENMP:BOOL=OFF

// Use MKL with TBB multithreading
MKL_WITH_TBB:BOOL=OFF

// OpenCL library is found
OPENCL_FOUND:BOOL=ON

// Where to create the platform-dependant cvconfig.h
OPENCV_CONFIG_FILE_INCLUDE_DIR:PATH=/opt/opencv-3.2.0/release

// Enable non-free algorithms
OPENCV_ENABLE_NONFREE:BOOL=OFF

// Where to look for additional OpenCV modules
OPENCV_EXTRA_MODULES_PATH:PATH=

// 
OPENCV_FORCE_PYTHON_LIBS:BOOL=OFF

// Treat warnings as errors
OPENCV_WARNINGS_ARE_ERRORS:BOOL=OFF

// Path to a file.
OPENEXR_INCLUDE_PATH:PATH=/usr/include/OpenEXR

// The directory containing a CMake configuration file for OpenCV_HAL.
OpenCV_HAL_DIR:PATH=OpenCV_HAL_DIR-NOTFOUND

// Path to Python interpretor
PYTHON2_EXECUTABLE:FILEPATH=/usr/bin/python2.7

// Python include dir
PYTHON2_INCLUDE_DIR:PATH=/usr/include/python2.7

// Python include dir 2
PYTHON2_INCLUDE_DIR2:PATH=

// Path to Python library
PYTHON2_LIBRARY:FILEPATH=/lib64/libpython2.7.so

// Path to Python debug
PYTHON2_LIBRARY_DEBUG:FILEPATH=

// Path to numpy headers
PYTHON2_NUMPY_INCLUDE_DIRS:PATH=/usr/lib64/python2.7/site-packages/numpy/core/include

// Where to install the python packages.
PYTHON2_PACKAGES_PATH:PATH=lib/python2.7/site-packages

// Path to Python interpretor
PYTHON3_EXECUTABLE:FILEPATH=/usr/bin/python3.4

// Python include dir
PYTHON3_INCLUDE_DIR:PATH=/usr/include/python3.4m

// Python include dir 2
PYTHON3_INCLUDE_DIR2:PATH=

// Path to Python library
PYTHON3_LIBRARY:FILEPATH=/lib64/libpython3.4m.so

// Path to Python debug
PYTHON3_LIBRARY_DEBUG:FILEPATH=

// Path to numpy headers
PYTHON3_NUMPY_INCLUDE_DIRS:PATH=/usr/lib64/python3.4/site-packages/numpy/core/include

// Where to install the python packages.
PYTHON3_PACKAGES_PATH:PATH=lib/python3.4/site-packages

// The directory containing a CMake configuration file for VTK.
VTK_DIR:PATH=VTK_DIR-NOTFOUND

// Include IEEE1394 support
WITH_1394:BOOL=ON

// Include Aravis GigE support
WITH_ARAVIS:BOOL=OFF

// Include Clp support (EPL)
WITH_CLP:BOOL=OFF

// Include NVidia Cuda Basic Linear Algebra Subprograms (BLAS) library support
WITH_CUBLAS:BOOL=OFF

// Include NVidia Cuda Runtime support
WITH_CUDA:BOOL=ON

// Include NVidia Cuda Fast Fourier Transform (FFT) library support
WITH_CUFFT:BOOL=ON

// Include Eigen2/Eigen3 support
WITH_EIGEN:BOOL=ON

// Include FFMPEG support
WITH_FFMPEG:BOOL=ON

// Include GDAL Support
WITH_GDAL:BOOL=OFF

// Include DICOM support
WITH_GDCM:BOOL=OFF

// Include Smartek GigE support
WITH_GIGEAPI:BOOL=OFF

// Include gPhoto2 library support
WITH_GPHOTO2:BOOL=ON

// Include Gstreamer support
WITH_GSTREAMER:BOOL=ON

// Enable Gstreamer 0.10 support (instead of 1.x)
WITH_GSTREAMER_0_10:BOOL=OFF

// Include GTK support
WITH_GTK:BOOL=ON

// Use GTK version 2
WITH_GTK_2_X:BOOL=OFF

// Include Intel IPP support
WITH_IPP:BOOL=ON

// Include Intel IPP_A support
WITH_IPP_A:BOOL=OFF

// Include JPEG2K support
WITH_JASPER:BOOL=ON

// Include JPEG support
WITH_JPEG:BOOL=ON

// Include Lapack library support
WITH_LAPACK:BOOL=ON

// Use libv4l for Video 4 Linux support
WITH_LIBV4L:BOOL=OFF

// Include Matlab support
WITH_MATLAB:BOOL=ON

// Include NVidia Video Decoding library support
WITH_NVCUVID:BOOL=OFF

// Include OpenCL Runtime support
WITH_OPENCL:BOOL=ON

// Include AMD OpenCL BLAS library support
WITH_OPENCLAMDBLAS:BOOL=ON

// Include AMD OpenCL FFT library support
WITH_OPENCLAMDFFT:BOOL=ON

// Include OpenCL Shared Virtual Memory support
WITH_OPENCL_SVM:BOOL=OFF

// Include ILM support via OpenEXR
WITH_OPENEXR:BOOL=ON

// Include OpenGL support
WITH_OPENGL:BOOL=OFF

// Include OpenMP support
WITH_OPENMP:BOOL=OFF

// Include OpenNI support
WITH_OPENNI:BOOL=OFF

// Include OpenNI2 support
WITH_OPENNI2:BOOL=OFF

// Include OpenVX support
WITH_OPENVX:BOOL=OFF

// Include PNG support
WITH_PNG:BOOL=ON

// Use pthreads-based parallel_for
WITH_PTHREADS_PF:BOOL=ON

// Include Prosilica GigE support
WITH_PVAPI:BOOL=OFF

// Build with Qt Backend support
WITH_QT:BOOL=OFF

// Include Intel TBB support
WITH_TBB:BOOL=OFF

// Include TIFF support
WITH_TIFF:BOOL=ON

// Include Unicap support (GPL)
WITH_UNICAP:BOOL=OFF

// Include Video 4 Linux support
WITH_V4L:BOOL=ON

// Include VA support
WITH_VA:BOOL=OFF

// Include Intel VA-API/OpenCL support
WITH_VA_INTEL:BOOL=OFF

// Include VTK library support (and build opencv_viz module eiher)
WITH_VTK:BOOL=ON

// Include WebP support
WITH_WEBP:BOOL=ON

// Include XIMEA cameras support
WITH_XIMEA:BOOL=OFF

// Include Xine support (GPL)
WITH_XINE:BOOL=OFF
```

### cmake
cmake -DBUILD_DOCS=ON \
-DBUILD_TESTS=OFF \
-DBUILD_PERF_TESTS=OFF \
-DBUILD_EXAMPLES=ON \
-DBUILD_JPEG=ON \
-DBUILD_PNG=ON \
-DBUILD_TBB=ON \
-DBUILD_TIFF=ON \
-DBUILD_ZLIB=ON \
-DBUILD_opencv_core=ON \
-DBUILD_opencv_ml=ON \
-DBUILD_opencv_python2=OFF \
-DBUILD_opencv_java=OFF \
-DWITH_FFMPEG=ON \
-DWITH_GSTREAMER=ON \
-DWITH_GTK=ON \
-DWITH_IPP=ON \
-DWITH_JPEG=ON \
-DWITH_LAPACK=OFF \
-DWITH_PNG=ON \
-DWITH_TBB=ON \
-DWITH_TIFF=ON \
-DWITH_CUDA=OFF \
-DWITH_OPENGL=ON \
-DWITH_OPENCL=ON \
-DWITH_EIGEN=ON \
-DWITH_V4L=ON \
-DWITH_VTK=OFF \
-DENABLE_AVX=ON \
-DPYTHON3_EXECUTABLE=$(which python3) \
-DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-DCMAKE_BUILD_TYPE=RELEASE \
-DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") ..

## Test OpenCV install

```
python3 -c "import cv2; print(cv2.__version__)"
```

## NOTE
### MKL, IPP, TBB, ATLAS, BLAS, LAPACK
关于矩阵库的选择，使用Python时安装numpy，如要高性能，可参考“[矩阵运算库blas, cblas, openblas, atlas, lapack, mkl之间有什么关系，在性能上区别大吗？](https://www.zhihu.com/question/27872849)”

## References
1. [Install OpenCV 3.1 and Python 2.7 on CentOS 7](http://www.computervisiononline.com/blog/install-opencv-31-and-python-27-centos-7)
2. [Compiling OpenCV with CUDA support](http://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support/)
3. [Compile OpenCV 3.2 for Anaconda Python 3.6, 3.5, 3.4 and 2.7](https://www.scivision.co/anaconda-python-opencv3/)
4. [OpenCV 3.1 Installation Guide on Ubuntu 16.04](https://github.com/BVLC/caffe/wiki/OpenCV-3.1-Installation-Guide-on-Ubuntu-16.04)
5. [Ubuntu 16.04: How to install OpenCV](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)
