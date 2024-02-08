# ---   CMAKE BUILDING  --- #
# ------------------------- #
ADD_LIBRARY(vl3dpp ${sources})


# List of all target libraries
set(
    VL3DPP_TARGET_LIBRARIES
    Python3::Python
    armadillo
    carma::carma
    ${PCL_LIBRARIES}
)

# LINK TARGET LIBRARIES
if(WIN32 OR MSVC)  # Windows compilation
    target_link_libraries(vl3dpp ${VL3DPP_TARGET_LIBRARIES})
else()  # Linux compilation
    target_link_libraries(vl3dpp ${VL3DPP_TARGET_LIBRARIES})
endif()
