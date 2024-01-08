/*
 * Provides functions wrapping VL3DPP to be easily called from the VL3D
 * python software.
 * More concretely, the functions here wrap data mining components.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <carma>
#include <armadillo>

namespace py = pybind11;

namespace vl3dpp::pymods {

py::array mine_smooth_feats(py::array X, py::array F){
    arma::Mat<double> Fhat = carma::arr_to_mat<double>(F);
    Fhat = Fhat + 10;
    return carma::mat_to_arr(Fhat, true);
}

}