#pragma once

#include "autodiff/forward/real.hpp"
#include "autodiff/forward/real/eigen.hpp"

#include <Eigen/Dense>
#include <Eigen/SparseCore>

namespace vurtis {
  // autodiff
  using VectorAD = autodiff::VectorXreal;
  using real = autodiff::real;
  // eigen
  using DiagonalMatrix = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;
  using SparseMatrixEigen = Eigen::SparseMatrix<double>;
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
}

