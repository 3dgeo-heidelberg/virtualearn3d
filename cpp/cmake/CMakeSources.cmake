# ---   CMAKE SOURCES   --- #
# ------------------------- #
# Register files
file(GLOB_RECURSE sources CONFIGURE_DEPENDS src/*.cpp include/*.hpp)

# VL3D++ include directories
set(VL3DPP_INCLUDE_DIRECTORIES
    "include/"
    "include/mining"
    "include/module"
)

include_directories(${VL3DPP_INCLUDE_DIRECTORIES})