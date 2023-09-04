#pragma once

#include "autodiff/forward/dual.hpp"
#include "autodiff/forward/dual/eigen.hpp"
#include <Eigen/Dense>
#include <Eigen/SparseCore>

namespace vurtis {
  // autodiff
  using MatrixAD = autodiff::MatrixXdual;
  using VectorAD = autodiff::VectorXdual;
  using ScalarAD = autodiff::dual;
  // eigen
  using SparseMatrixEigen = Eigen::SparseMatrix<double>;
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;


  namespace utils {

      template<typename Scalar, typename StorageIndex>
      void sparse_stack_h_inplace(
          Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>& left,
          const Eigen::SparseMatrix<Scalar, Eigen::ColMajor, StorageIndex>& right)
      {
        assert(left.rows() == right.rows());

        const StorageIndex leftcol = (StorageIndex)left.cols();
        const StorageIndex leftnz = (StorageIndex)left.nonZeros();

        left.conservativeResize(left.rows(), left.cols() + right.cols());
        left.resizeNonZeros(left.nonZeros() + right.nonZeros());

        std::copy(right.innerIndexPtr(), right.innerIndexPtr() + right.nonZeros(), left.innerIndexPtr() + leftnz);
        std::copy(right.valuePtr(), right.valuePtr() + right.nonZeros(), left.valuePtr() + leftnz);
        std::transform(right.outerIndexPtr(), right.outerIndexPtr() + right.cols() + 1, left.outerIndexPtr() + leftcol, [&](StorageIndex i) { return i + leftnz; });
      }



      static std::vector<Eigen::Triplet<double>> to_triplets(Eigen::SparseMatrix<double> & M){
        std::vector<Eigen::Triplet<double>> v;
        for(int i = 0; i < M.outerSize(); i++)
          for(typename Eigen::SparseMatrix<double>::InnerIterator it(M,i); it; ++it)
            v.emplace_back(it.row(),it.col(),it.value());
        return v;
      }

  }
}

