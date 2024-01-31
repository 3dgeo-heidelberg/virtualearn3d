# Container for building the environment
FROM ubuntu:22.04 as ubuntu

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC  apt-get install -y curl \
    cmake build-essential wget \
    mesa-common-dev libglu1-mesa-dev libeigen3-dev python3.10 python3-pip \
    python3.10-dev pypy libpdal-dev gdal-bin libgdal-dev cython3 \
    libboost-all-dev libflann-dev libvtk9-dev libqhull-dev libpcl-dev git


# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY requirements.txt /app/requirements.txt
COPY cpp /app/cpp

WORKDIR /app

RUN pip install -r requirements.txt
# Install Python packages

RUN mkdir -p /app/cpp/lib 
WORKDIR /app/cpp/lib


# Fetch armadillo
RUN wget https://sourceforge.net/projects/arma/files/armadillo-12.6.7.tar.xz && tar -xf armadillo-12.6.7.tar.xz && mv armadillo-12.6.7 armadillo

# Fetch carma
RUN git clone https://github.com/RUrlus/carma

# Fetch pcl
RUN wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.13.0/source.tar.gz && tar -xf source.tar.gz \
    && cd pcl && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && cmake --build . -j4 && make -j4 install  \
    && cd ../..




RUN mkdir /app/cpp/build

WORKDIR /app/cpp/build

RUN python3 -c "import numpy"

RUN ls /app/cpp/lib/pcl/build/lib && which python3 && python3 --version \
&& cmake -DPCL_LIBRARIES=/app/cpp/lib/pcl/build/lib -DPCL_1.13_INCLUDE_DIR=/app/cpp/lib/pcl/build/include ..

RUN LDFLAGS="-L/app/cpp/lib/pcl/build/lib" CFLAGS="-I/app/cpp/lib/pcl/build/include" make

# Copy the built binaries to the out folder
RUN ls /app/cpp/build && mkdir /app/out && \
    cp /app/cpp/build/libvl3dpp.so /app/out && \
    cp /app/cpp/build/pyvl3dpp.cpython-310-x86_64-linux-gnu.so /app/out/pyvl3dpp.so


