#pragma once

#include "autodiff/forward/dual.hpp"
#include "autodiff/forward/dual/eigen.hpp"
#include <Eigen/Dense>
#include <Eigen/SparseCore>

namespace vurtis {
  // autodiff
  using VectorAD = autodiff::VectorXdual;
  using ScalarAD = autodiff::dual;
  // eigen
  using SparseMatrixEigen = Eigen::SparseMatrix<double>;
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
}

