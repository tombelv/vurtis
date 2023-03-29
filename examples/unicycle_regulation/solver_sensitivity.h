//
// Created by tom on 29/03/23.
//

#ifndef EXAMPLE_SOLVER_SENSITIVITY_H
#define EXAMPLE_SOLVER_SENSITIVITY_H


#include "vurtis/vurtis.h"
namespace vurtis {
/*
    VectorAD Solver::ConstraintStack(const Vector &w_guess, VectorAD &w, VectorAD &params) {

      VectorAD g(nx_ *(N_+1) +nh_ * N_ + nh_e_);
      g.setZero();

      VectorAD dx = w.head(nx_ * (N_ + 1));
      VectorAD du = w.tail(nu_ * N_);

      g.head(nx_) = dx.head(nx_) + w_guess.head(nx_) - params.head(nx_);

      for (int i = 0; i < N_; ++i) {
        VectorAD xk_guess = w_guess.segment(i * nx_, nx_);
        VectorAD uk_guess = w_guess.segment(nx_ * (N_ + 1) + i * nu_, nu_);
        VectorAD model_params = params.tail(params.size() - nx_);
        MatrixAD dF = model_->dFdxu(xk_guess, uk_guess, model_params);
        g.segment((i + 1) * nx_, nx_) = dx.segment((i + 1) * nx_, nx_) + w_guess.segment((i + 1) * nx_, nx_)
                                        - dF.leftCols(nx_) * dx.segment(i * nx_, nx_)
                                        - dF.middleCols(nx_, nu_) * du.segment(i * nu_, nu_)
                                        - model_->step(xk_guess, uk_guess, model_params);

        MatrixAD dC = model_->dConstrdxu(xk_guess, uk_guess, model_params, parameters_.col(i));
        g.segment(nx_ * (N_ + 1) + i * nh_, nh_) = dC.leftCols(nx_) * dx.segment(i * nx_, nx_)
                                                   + dC.middleCols(nx_, nu_) * du.segment(i * nu_, nu_)
                                                   + model_->Constraint(xk_guess, uk_guess, parameters_.col(i));
      }

      return g;
    }

/// This computes the constraint matrix in an AD-enabled way
    VectorAD Solver::dLdw(const Vector &w_guess, VectorAD &params) {
      MatrixAD res(nx_ *(N_
      +1) +nh_ * N_ + nh_e_, w_guess.size());
      res.setZero();

      res.topLeftCorner(nx_ * (N_ + 1), nx_ * (N_ + 1)) = MatrixAD::Identity(nx_ * (N_ + 1), nx_ * (N_ + 1));

      for (int i = 0; i < N_; ++i) {
        VectorAD xk_guess = w_guess.segment(i * nx_, nx_);
        VectorAD uk_guess = w_guess.segment(nx_ * (N_ + 1) + i * nu_, nu_);
        VectorAD model_params = params.tail(params.size() - nx_);

        MatrixAD dF = model_->dFdxu(xk_guess, uk_guess, model_params);
        MatrixAD dC = model_->dConstrdxu(xk_guess, uk_guess, model_params, parameters_.col(i));

        res.block((i + 1) * nx_, i * nx_, nx_, nx_) = -dF.leftCols(nx_);
        res.block((i + 1) * nx_, nx_ * (N_ + 1) + i * nu_, nx_, nu_) = -dF.rightCols(nu_);

        res.block(nx_ * (N_ + 1) + i * nh_, i * nx_, nh_, nx_) = dC.leftCols(nx_);
        res.block(nx_ * (N_ + 1) + i * nh_, nx_ * (N_ + 1) + i * nu_, nh_, nu_) = dC.rightCols(nu_);
      }

      return this->GetMultipliers().transpose() * res;
    }


    void Solver::ComputeSensitivity() {

      Vector w_guess(dw_.size());
      w_guess << x_guess_, u_guess_;

      VectorAD w(dw_);

      VectorAD param(nx_ + model_->GetModelParams().size());
      param << x_current_, model_->GetModelParams();

      // Compute the set of active inequality constraints by checking the multipliers
      Vector multipliers = this->GetMultipliers();
      std::vector<int> active_constraints;
      active_constraints.reserve(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);
      for (int i = 0; i < nx_ * (N_ + 1); ++i) active_constraints.push_back(i);
      for (int i = nx_ * (N_ + 1); i < nx_ * (N_ + 1) + nh_ * N_ + nh_e_; ++i) {
        if (abs(multipliers(i)) > 1e-6) active_constraints.push_back(i);
      }

      std::vector<Eigen::Triplet<double>> tripletList;
      tripletList.reserve(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);

      for (int i = 0; i < active_constraints.size(); ++i) {
        tripletList.push_back(Eigen::Triplet<double>(i, active_constraints[i], 1.0));
      }


      SparseMatrixEigen active_proj_matrix_sparse;

      active_proj_matrix_sparse.resize(active_constraints.size(), constraint_matrix_.rows());
      active_proj_matrix_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
      active_proj_matrix_sparse.makeCompressed();

      SparseMatrixEigen Jx_sparse = active_proj_matrix_sparse * constraint_matrix_;

      SparseMatrixEigen KKT_top = 0.5*hessian_matrix_;
      SparseMatrixEigen Jx_sparse_transp = Jx_sparse.transpose();
      utils::sparse_stack_h_inplace(KKT_top, Jx_sparse_transp);

      SparseMatrixEigen KKT_partial(hessian_matrix_.rows() + Jx_sparse.rows(),hessian_matrix_.rows() + Jx_sparse.rows());
      auto KKT_top_triplet = utils::to_triplets(KKT_top);
      KKT_partial.setFromTriplets(KKT_top_triplet.begin(), KKT_top_triplet.end());
      KKT_partial.makeCompressed();

      //KKT_top.resize(hessian_matrix_.rows() + Jx.rows(),hessian_matrix_.rows() + Jx.rows());

      SparseMatrixEigen KKT_sparse(hessian_matrix_.rows() + Jx_sparse.rows(), hessian_matrix_.rows() + Jx_sparse.rows());
      KKT_sparse =  KKT_partial + SparseMatrixEigen(KKT_partial.transpose());


*/
/*      Matrix KKT_matrix = Eigen::MatrixXd::Zero(H.rows() + Jx.rows(), H.rows() + Jx.rows());
      KKT_matrix.topLeftCorner(H.rows(), H.rows()) = H;
      KKT_matrix.bottomLeftCorner(Jx.rows(), Jx.cols()) = Jx;
      KKT_matrix.topRightCorner(Jx.cols(), Jx.rows()) = Jx.transpose();*//*



      Matrix dLwdp = jacobian([&](const Vector &w_guess, VectorAD &params) { return dLdw(w_guess, params); },
                              wrt(param), at(w_guess, param));

      Matrix dCdp = active_proj_matrix_sparse * jacobian(
          [&](const Vector &w_guess, VectorAD &w, VectorAD &params) { return ConstraintStack(w_guess, w, params); },
          wrt(param), at(w_guess, w, param));


      Matrix rightTerm(dLwdp.rows() + dCdp.rows(), param.size());
      rightTerm << dLwdp, dCdp;

      Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
      solver.analyzePattern(KKT_sparse);
      solver.factorize(KKT_sparse);
      Matrix paramSensitivity = solver.solve(-rightTerm);

      //Matrix paramSensitivity = KKT_matrix.partialPivLu().solve(-rightTerm);

      //Matrix paramSensitivity = -KKT_matrix.inverse()*(rightTerm);
      //std::cout << paramSensitivity << std::endl;

      sensitivity_ << paramSensitivity.topRows(nx_ * (N_ + 1)).transpose().format(CSVFormat) << "\n";


    }
*/



class SolverParametric : public vurtis::Solver {
public:

        using Solver::Solver;

    VectorAD ConstraintStack(const Vector &w_guess, VectorAD &w, VectorAD &params) {

      const int nx_ = model_->nx_;
      const int nu_ = model_->nu_;
      const int nh_ = model_->nh_;
      const int nh_e_ = model_->nh_e_;
      const int N_ = this->GetN();

      VectorAD g(nx_ *(N_+1) +nh_ * N_ + nh_e_);
      g.setZero();

      VectorAD dx = w.head(nx_ * (N_ + 1));
      VectorAD du = w.tail(nu_ * N_);

      g.head(nx_) = dx.head(nx_) + w_guess.head(nx_) - params.head(nx_);

      for (int i = 0; i < N_; ++i) {
        VectorAD xk_guess = w_guess.segment(i * nx_, nx_);
        VectorAD uk_guess = w_guess.segment(nx_ * (N_ + 1) + i * nu_, nu_);
        VectorAD model_params = params.tail(params.size() - nx_);
        MatrixAD dF = model_->dFdxu(xk_guess, uk_guess, model_params);
        g.segment((i + 1) * nx_, nx_) = dx.segment((i + 1) * nx_, nx_) + w_guess.segment((i + 1) * nx_, nx_)
                                        - dF.leftCols(nx_) * dx.segment(i * nx_, nx_)
                                        - dF.middleCols(nx_, nu_) * du.segment(i * nu_, nu_)
                                        - model_->step(xk_guess, uk_guess, model_params);

        MatrixAD dC = model_->dConstrdxu(xk_guess, uk_guess, model_params, GetNlpParams().col(i));
        g.segment(nx_ * (N_ + 1) + i * nh_, nh_) = dC.leftCols(nx_) * dx.segment(i * nx_, nx_)
                                                   + dC.middleCols(nx_, nu_) * du.segment(i * nu_, nu_)
                                                   + model_->Constraint(xk_guess, uk_guess, GetNlpParams().col(i));
      }

      return g;
    }

/// This computes the constraint matrix in an AD-enabled way
    VectorAD dLdw(const Vector &w_guess, VectorAD &params) {
      const int nx_ = model_->nx_;
      const int nu_ = model_->nu_;
      const int nh_ = model_->nh_;
      const int nh_e_ = model_->nh_e_;
      const int N_ = this->GetN();
      MatrixAD res(nx_ *(N_
      +1) +nh_ * N_ + nh_e_, w_guess.size());
      res.setZero();

      res.topLeftCorner(nx_ * (N_ + 1), nx_ * (N_ + 1)) = MatrixAD::Identity(nx_ * (N_ + 1), nx_ * (N_ + 1));

      for (int i = 0; i < N_; ++i) {
        VectorAD xk_guess = w_guess.segment(i * nx_, nx_);
        VectorAD uk_guess = w_guess.segment(nx_ * (N_ + 1) + i * nu_, nu_);
        VectorAD model_params = params.tail(params.size() - nx_);

        MatrixAD dF = model_->dFdxu(xk_guess, uk_guess, model_params);
        MatrixAD dC = model_->dConstrdxu(xk_guess, uk_guess, model_params, GetNlpParams().col(i));

        res.block((i + 1) * nx_, i * nx_, nx_, nx_) = -dF.leftCols(nx_);
        res.block((i + 1) * nx_, nx_ * (N_ + 1) + i * nu_, nx_, nu_) = -dF.rightCols(nu_);

        res.block(nx_ * (N_ + 1) + i * nh_, i * nx_, nh_, nx_) = dC.leftCols(nx_);
        res.block(nx_ * (N_ + 1) + i * nh_, nx_ * (N_ + 1) + i * nu_, nh_, nu_) = dC.rightCols(nu_);
      }

      return this->GetMultipliers().transpose() * res;
    }


    void ComputeSensitivity(Matrix & sensitivity) {

      const int nx_ = model_->nx_;
      const int nu_ = model_->nu_;
      const int nh_ = model_->nh_;
      const int nh_e_ = model_->nh_e_;
      const int N_ = this->GetN();

      VectorAD w(GetSolutionDelta());

      Vector w_guess(w.size());
      w_guess << GetStateTrajectory(), GetInputTrajectory();



      VectorAD param(nx_ + model_->GetModelParams().size());
      param << GetCurrentState(), model_->GetModelParams();

      // Compute the set of active inequality constraints by checking the multipliers
      Vector multipliers = this->GetMultipliers();
      std::vector<int> active_constraints;
      active_constraints.reserve(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);
      for (int i = 0; i < nx_ * (N_ + 1); ++i) active_constraints.push_back(i);
      for (int i = nx_ * (N_ + 1); i < nx_ * (N_ + 1) + nh_ * N_ + nh_e_; ++i) {
        if (abs(multipliers(i)) > 1e-6) active_constraints.push_back(i);
      }

      std::vector<Eigen::Triplet<double>> tripletList;
      tripletList.reserve(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);

      for (int i = 0; i < active_constraints.size(); ++i) {
        tripletList.push_back(Eigen::Triplet<double>(i, active_constraints[i], 1.0));
      }

      SparseMatrixEigen H = this->GetHessianMatrix();
      SparseMatrixEigen G = this->GetConstraintMatrix();

      SparseMatrixEigen active_proj_matrix_sparse;

      active_proj_matrix_sparse.resize(active_constraints.size(), G.rows());
      active_proj_matrix_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
      active_proj_matrix_sparse.makeCompressed();

      SparseMatrixEigen G_active = active_proj_matrix_sparse * G;

      SparseMatrixEigen KKT_top = 0.5*H;
      SparseMatrixEigen G_active_transp = G_active.transpose();
      utils::sparse_stack_h_inplace(KKT_top, G_active_transp);

      SparseMatrixEigen KKT_partial(H.rows() + G_active.rows(),H.rows() + G_active.rows());
      auto KKT_top_triplet = utils::to_triplets(KKT_top);
      KKT_partial.setFromTriplets(KKT_top_triplet.begin(), KKT_top_triplet.end());
      KKT_partial.makeCompressed();

      //KKT_top.resize(H.rows() + Jx.rows(),H.rows() + Jx.rows());

      SparseMatrixEigen KKT_sparse(H.rows() + G_active.rows(), H.rows() + G_active.rows());
      KKT_sparse =  KKT_partial + SparseMatrixEigen(KKT_partial.transpose());


/*      Matrix KKT_matrix = Eigen::MatrixXd::Zero(H.rows() + Jx.rows(), H.rows() + Jx.rows());
      KKT_matrix.topLeftCorner(H.rows(), H.rows()) = H;
      KKT_matrix.bottomLeftCorner(Jx.rows(), Jx.cols()) = Jx;
      KKT_matrix.topRightCorner(Jx.cols(), Jx.rows()) = Jx.transpose();*/


      Matrix dLwdp = jacobian([&](const Vector &w_guess, VectorAD &params) { return dLdw(w_guess, params); },
                              wrt(param), at(w_guess, param));

      Matrix dCdp = active_proj_matrix_sparse * jacobian(
          [&](const Vector &w_guess, VectorAD &w, VectorAD &params) { return ConstraintStack(w_guess, w, params); },
          wrt(param), at(w_guess, w, param));


      Matrix rightTerm(dLwdp.rows() + dCdp.rows(), param.size());
      rightTerm << dLwdp, dCdp;

      Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
      solver.analyzePattern(KKT_sparse);
      solver.factorize(KKT_sparse);

      sensitivity = solver.solve(-rightTerm);

      //Matrix paramSensitivity = KKT_matrix.partialPivLu().solve(-rightTerm);

      //Matrix paramSensitivity = -KKT_matrix.inverse()*(rightTerm);
      //std::cout << paramSensitivity << std::endl;




    }
};

}

#endif //EXAMPLE_SOLVER_SENSITIVITY_H
