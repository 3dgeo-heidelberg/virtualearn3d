# ---   CMAKE PYBIND11   --- #
# -------------------------- #

# Add PyBind11
find_package(Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS})
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
pybind11_add_module(pyvl3dpp MODULE src/module/vl3dpp.cpp)
target_link_libraries(pyvl3dpp PRIVATE ${Python3_LIBRARIES})
install(TARGETS pyvl3dpp DESTINATION .)
