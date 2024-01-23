# ---   CMAKE LIBRARIES   --- #
# --------------------------- #

# Armadillo
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/armadillo)  # Use armadillo from lib
    # Include from lib directory
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/armadillo)
    set(ARMADILLO_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib/armadillo/include)
else()  # Try to find already installed armadillo
    find_package(Armadillo REQUIRED)
endif()

# Armadillo with custom LAPACK
if(LAPACK_LIB)
    # Add custom lapack library to armadillo if specified
    set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARIES} ${LAPACK_LIB})
endif()

# Include directories
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Report included directories and libraries
#message("Armadillo root: " ${ARMADILLO_ROOT_DIR})
message("Armadillo include: " ${ARMADILLO_INCLUDE_DIRS})
#message("Armadillo libraries: " ${ARMADILLO_LIBRARIES})





# Carma
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/carma)  # Use carma from lib
    # Include from lib directory
    add_library(armadillo:armadillo ALIAS armadillo)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/carma/)
    set(carma_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/carma/include)
    set(carma_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib/carma/include)
else()  # Try to find already installed Carma
    find_package(Carma CONFIG REQUIRED)
endif()

# Include directories
include_directories(${carma_INCLUDE_DIR})
include_directories(${carma_INCLUDE_DIRS})

# Report included directories and libraries
message("Carma found: " ${carma_FOUND})
message("Carma include dir: " ${carma_INCLUDE_DIR})
message("Carma include dirs: " ${carma_INCLUDE_DIRS})





# PCL (PointCloudLibrary)
# TODO Rethink : Also install Eigen because PCL needs it
# TODO Rethink : FLANN, Qhull, and OpenMP might be necessary too
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/pcl)  # Use PCL from lib
    # TODO Rethink : The disabled modules are considered when building the first time
    set(PCL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib/pcl/install/include)
    set(BUILD_common 1)
    set(BUILD_kdtree 1)
    set(BUILD_octree 1)
    set(BUILD_search 1)
    set(BUILD_sample_consensus 0)
    set(BUILD_filters 0)
    set(BUILD_2d 0)
    set(BUILD_geometry 0)
    set(BUILD_io 0)
    set(BUILD_features 0)
    set(BUILD_ml 0)
    set(BUILD_segmentation 0)
    set(BUILD_registration 0)
    set(BUILD_keypoints 0)
    set(BUILD_tracking 0)
    set(BUILD_recognition 0)
    set(BUILD_stereo 0)
    set(BUILD_tools 0)
    set(BUILD_visualization 0)
    set(BUILD_surface 0)
    set(WITH_VTK 0)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/pcl)
else()  # Try to find already installed PCL
    find_package(PCL REQUIRED 1.13)
endif()
