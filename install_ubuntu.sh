#!/bin/bash - 
#===============================================================================
#
#          FILE: install_ubuntu.sh
# 
#         USAGE: ./install_ubuntu.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: YOUR NAME (), 
#  ORGANIZATION: 
#       CREATED: 05/27/2017 17:03
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

apt-get install --assume-yes build-essential cmake git  
apt-get install --assume-yes build-essential pkg-config unzip ffmpeg qtbase5-dev  
apt-get install --assume-yes libopencv-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev  
apt-get install --assume-yes libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev  
apt-get install --assume-yes libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev  
apt-get install --assume-yes libvorbis-dev libxvidcore-dev v4l-utils  
pip3 install numpy  

cd /opt
git clone https://github.com/opencv/opencv.git  
git clone https://github.com/opencv/opencv_contrib.git  

cd opencv_contrib  
git checkout 3.2.0  
cd ../opencv  
git checkout 3.2.0  
mkdir release  
cd release  

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

make -j$(($(nproc) + 1))  
#make install   
#ldconfig  
