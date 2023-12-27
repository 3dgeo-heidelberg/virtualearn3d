#include <pybind11/pybind11.h>
#include <module/DataMiningModule.hpp>
#include <string>

std::string get_hello_world() {
    return "HELLO WORLD";
}

PYBIND11_MODULE(pyvl3dpp, m){
    // Hello world
    m.def(
        "get_hello_world",
        &get_hello_world,
        "Return the HELLO WORLD string"
    );




    // ***  DATA MINING MODULES  *** //
    // ***************************** //
    m.def(
        "mine_smooth_feats",                                // Function name
        &vl3dpp::pymods::mine_smooth_feats,                 // Function wrapper
        "Mine smooth features"                              // Description
    );




}