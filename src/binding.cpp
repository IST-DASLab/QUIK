#include <pybind11/pybind11.h>

#include "matmul/matmul.h"
#include "symmetric/symmetric.h"
#include "asymmetric/asymmetric.h"

PYBIND11_MODULE(_C, mod) {
  QUIK::matmul::buildSubmodule(mod);
  QUIK::symmetric::buildSubmodule(mod);
  QUIK::asymmetric::buildSubmodule(mod);
}
