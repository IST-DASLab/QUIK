#pragma once
#include <pybind11/pybind11.h>

namespace QUIK::matmul {
void buildSubmodule(pybind11::module &mod);
}
