#pragma once
// osqp-eigen
#include "OsqpEigen/OsqpEigen.h"

#include "vurtis/cost_base.h"
#include "vurtis/model_base.h"
#include "utils.h"

namespace vurtis {

class Solver {
  friend OsqpEigen::Solver;

 private:

  size_t nx_;
  size_t nu_;
  size_t nz_;
  size_t nh_;
  size_t nh_e_;

  size_t num_parameters_;
  size_t N_;

  Matrix Ad_list_;
  Matrix Bd_list_;
  Matrix Cd_list_;
  Matrix Cd_e;
  Matrix Dd_list_;

  std::shared_ptr<ModelBase> model_;
  std::shared_ptr<CostBase> cost_;

  Vector x_guess_;
  Vector u_guess_;

  Matrix parameters_;



  Vector x_current_;



  // Cost function
  //Matrix gradient_cost;

  Matrix R_list_;
  Matrix dRdx_list;
  Matrix dRdu_list;

  SparseMatrixEigen hessian_matrix_;
  SparseMatrixEigen constraint_matrix_;
  Vector gradient_;
  Vector residual_x_;
  Vector residual_h_;
  Vector lower_bound_;
  Vector upper_bound_;

  OsqpEigen::Solver QPsolver_;

 public:

  Solver(const std::shared_ptr<ModelBase> model, const std::shared_ptr<CostBase> cost, const ProblemInit &problemParams) : model_{model}, cost_{cost} {

    x_current_ = problemParams.x0;

    nx_ = problemParams.nx;
    nu_ = problemParams.nu;
    nz_ = problemParams.nz;
    nh_ = problemParams.nh;
    nh_e_ = problemParams.nh_e;

    N_ = problemParams.N;

    num_parameters_ = problemParams.num_parameters;

    InitMatricesToZero();

    residual_x_.resize(nx_ * N_);
    residual_h_.resize(nh_ * N_ + nh_e_);

    hessian_matrix_.resize(nx_ * (N_ + 1) + nu_ * N_, nx_ * (N_ + 1) + nu_ * N_);
    constraint_matrix_.resize(nx_ * (N_ + 1) + nh_ * N_ + nh_e_, nx_ * (N_ + 1) + nu_ * N_);

    lower_bound_.resize(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);
    upper_bound_.resize(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);


    InitSolutionGuess();

    InitHessian();

    InitGradient();

    InitConstraintMatrix();

    ComputeBounds();

    SetupQPsolver();

  }

  void InitMatricesToZero() {

    parameters_ = Matrix::Zero(num_parameters_, N_ + 1);

    Ad_list_ = Matrix::Zero(nx_, nx_ * N_);
    Bd_list_ = Matrix::Zero(nx_, nu_ * N_);
    Cd_list_ = Matrix::Zero(nh_, nx_ * N_);
    Cd_e = Matrix::Zero(nh_e_, nx_);
    Dd_list_ = Matrix::Zero(nh_, nu_ * N_);

    R_list_ = Matrix::Zero(nz_, N_);
    dRdx_list = Matrix::Zero(nz_, nx_ * N_);
    dRdu_list = Matrix::Zero(nz_, nu_ * N_);

    gradient_ = Vector::Zero(nx_ * (N_ + 1) + nu_ * N_);

  }

  void InitHessian() {


    std::vector<Eigen::Triplet<double>> tripletList;

    ComputeCost();

    tripletList.reserve(nx_ * (N_ + 1) + nu_ * N_);


    // Build the N_ state blocks (dim: nx_*N_)
    // Outer loop iterates over blocks, inner loop over Q
    // e.g. (nx_ = 3) i=0,3,6,9... ii=0,1,2,0,1,2,0,1,2...
    for (size_t i = 0; i < nx_ * N_; i = i + nx_) {
      Matrix dRdx_pwise = (dRdx_list.middleCols(i,nx_)).transpose()*dRdx_list.middleCols(i,nx_);
      for (size_t row_idx = 0; row_idx < nx_; ++row_idx) {
        for (size_t col_idx = 0; col_idx < nx_; ++col_idx)
          tripletList.push_back(Eigen::Triplet<double>(i + row_idx, i + col_idx, dRdx_pwise(row_idx, col_idx)));
      }
    }

    // !! For now the end cost is set to zero !!
    // Add the end cost_ block (dim: nx_)
    { size_t i =  nx_ * N_;
      for (size_t row_idx = 0; row_idx < nx_; ++row_idx) {
        for (size_t col_idx = 0; col_idx < nx_; ++col_idx)
          tripletList.push_back(Eigen::Triplet<double>(i + row_idx, i + col_idx, 0.0));
      }
    }



    // Build the N_ input (dim: nu_*N_)
    for (size_t i = nx_ * (N_ + 1); i < nx_ * (N_ + 1) + nu_ * N_; i = i + nu_) {
      Matrix dRdu_pwise = (dRdu_list.middleCols(i-nx_ * (N_ + 1),nu_)).transpose()*dRdu_list.middleCols(i-nx_ * (N_ + 1),nu_);
      for (size_t row_idx = 0; row_idx < nu_; ++row_idx) {
        for (size_t col_idx = 0; col_idx < nu_; ++col_idx)
          tripletList.push_back(Eigen::Triplet<double>(i + row_idx, i + col_idx, dRdu_pwise(row_idx, col_idx)));
      }
    }

    hessian_matrix_.setFromTriplets(tripletList.begin(), tripletList.end());

    hessian_matrix_.makeCompressed();


  }



  void InitGradient() {

    for(size_t i = 0; i < N_; ++i) {
      Matrix dR(nz_,nx_+nu_);
      dR << dRdx_list.middleCols(i*nx_,nx_), dRdu_list.middleCols(i*nu_,nu_);
      Vector grad = (R_list_.col(i)).transpose()*dR;
      gradient_.segment(i * nx_, nx_) = grad.head(nx_);
      gradient_.segment(nx_*(N_+1) + i * nu_, nu_) = grad.tail(nu_);
    }
  }

  void InitConstraintMatrix() {
    std::vector<Eigen::Triplet<double>> tripletList;

    tripletList.reserve((nx_ + 1 + nh_) * (nx_ * N_) + (1 + nh_e_) * nx_ + (nx_ + nh_) * nu_ * N_);

    // Build identity block
    for (size_t i = 0; i < nx_ * (N_ + 1); ++i)
      tripletList.push_back(Eigen::Triplet<double>(i, i, 1.0));

    for (size_t block = 0; block < N_; ++block) {
      for (size_t col = block * nx_; col < (block + 1) * nx_; ++col) {

        for (size_t row = (block + 1) * nx_; row < (block + 2) * nx_; ++row)
          tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));

        for (size_t row = nx_ * (N_ + 1) + block * nh_; row < nx_ * (N_ + 1) + (block + 1) * nh_; ++row)
          tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));
      }
    }

    for (size_t block = 0; block < N_; ++block) {
      for (size_t col = nx_ * (N_ + 1) + block * nu_; col < nx_ * (N_ + 1) + (block + 1) * nu_; ++col) {

        for (size_t row = (block + 1) * nx_; row < (block + 2) * nx_; ++row)
          tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));

        for (size_t row = nx_ * (N_ + 1) + block * nh_; row < nx_ * (N_ + 1) + (block + 1) * nh_; ++row)
          tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));
      }
    }

    for (size_t col = nx_ * N_; col < nx_ * (N_ + 1); ++col) {
      for (size_t row = nx_ * (N_ + 1) + nh_ * N_; row < nx_ * (N_ + 1) + nh_ * N_ + nh_e_; ++row)
        tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));
    }

    constraint_matrix_.setFromTriplets(tripletList.begin(), tripletList.end());
    constraint_matrix_.makeCompressed();

    ComputeSensitivitiesAndResiduals();

    UpdateConstraintMatrix();

  }

  void InitSolutionGuess() {
    x_guess_ = cost_->x_ref_;
    u_guess_ = cost_->u_ref_;
  }

  void UpdateHessian() {

    Vector nonzero(nx_*nx_*(N_+1) + nu_*nu_*N_);

    for (size_t i = 0; i < nx_ * N_; i = i + nx_) {
      Matrix dRdx_pwise = (dRdx_list.middleCols(i,nx_)).transpose()*dRdx_list.middleCols(i,nx_);
      for (size_t ii = 0; ii < nx_; ++ii)
        nonzero.segment(nx_*(i + ii), nx_) = dRdx_pwise.col(ii);
    }

    // For now the terminal cost is zero
    { size_t i = nx_*N_;
      for (size_t ii = 0; ii < nx_; ++ii)
        nonzero.segment(nx_*(i + ii), nx_) = Vector::Zero(nx_);
    }

    size_t input_start_coeff = nx_*nx_*(N_+1);

    for (size_t i = 0; i < nu_ * N_; i = i + nu_) {
      Matrix dRdu_pwise = (dRdu_list.middleCols(i,nu_)).transpose()*dRdu_list.middleCols(i,nu_);
      for (size_t ii = 0; ii < nu_; ++ii)
        nonzero.segment(nu_*(i + ii)+input_start_coeff, nu_) = dRdu_pwise.col(ii);
    }

    hessian_matrix_.coeffs() = nonzero;

    //std::cout << nonzero << std::endl;

  }

  void UpdateGradient() {
    //Vector tracking_error{nx_ * (N_ + 1) + nu_ * N_};
    //tracking_error << (x_guess_ - cost_->x_ref_), (u_guess_ - cost_->u_ref_);

    //gradient_ = hessian_matrix_ * tracking_error;
    for(size_t i = 0; i < N_; ++i) {
      Matrix dR(nz_,nx_+nu_);
      dR << dRdx_list.middleCols(i*nx_,nx_), dRdu_list.middleCols(i*nu_,nu_);
      Vector grad = (R_list_.col(i)).transpose()*dR;
      gradient_.segment(i * nx_, nx_) = grad.head(nx_);
      gradient_.segment(nx_*(N_+1) + i * nu_, nu_) = grad.tail(nu_);
    }
  }



  void UpdateConstraintMatrix() {

    Vector nonzero((1 + nx_ + nh_) * (nx_ * N_) + (1 + nh_e_) * nx_ + (nx_ + nh_) * nu_ * N_);
    for (size_t i = 0; i < nx_ * (N_); ++i) {
      nonzero(i * (nx_ + 1 + nh_)) = 1;
      nonzero.segment(i * (nx_ + 1 + nh_) + 1, nx_) = -Ad_list_.col(i);
      nonzero.segment(i * (nx_ + 1 + nh_) + 1 + nx_, nh_) = Cd_list_.col(i);
    }

    //nonzero.segment((nx_+1+nh_)*(nx_*N_),nx_) << Vector::Ones(nx_);
    for (size_t i = 0; i < nx_; ++i)
      nonzero.segment(i * (1 + nh_e_) + (nx_ + 1 + nh_) * (nx_ * N_), 1 + nh_e_) << 1, Cd_e.col(i);

    size_t input_start_coeff = (nx_ + 1 + nh_) * (nx_ * N_) + (1 + nh_e_) * nx_;

    for (size_t i = 0; i < nu_ * N_; ++i) {
      nonzero.segment(i * (nx_ + nh_) + input_start_coeff, nx_) = -Bd_list_.col(i);
      nonzero.segment(i * (nx_ + nh_) + input_start_coeff + nx_, nh_) = Dd_list_.col(i);
    }

    constraint_matrix_.coeffs() = nonzero;

  }

  void ComputeCost() {
    for (size_t idx = 0; idx < N_; ++idx) {
      VectorAD state = x_guess_.segment(idx * nx_, nx_);
      VectorAD input = u_guess_.segment(idx * nu_, nu_);
      Vector state_ref = cost_->x_ref_.segment(idx * nx_, nx_);
      Vector input_ref = cost_->u_ref_.segment(idx * nu_, nu_);

      Matrix dR = cost_->GradientCost(state,input,state_ref,input_ref);
      dRdx_list.middleCols(idx * nx_, nx_) = dR.leftCols(nx_);
      dRdu_list.middleCols(idx * nu_, nu_) = dR.rightCols(nu_);

      R_list_.col(idx) = cost_->cost_eval_.cast<double>();
    }

  }

  void ComputeSensitivitiesAndResiduals() {
    for (size_t idx = 0; idx < N_; ++idx) {
      VectorAD state = x_guess_.segment(idx * nx_, nx_);
      VectorAD state_next = x_guess_.segment((idx + 1) * nx_, nx_);
      VectorAD input = u_guess_.segment(idx * nu_, nu_);
      Vector params = parameters_.col(idx);

      Ad_list_.middleCols(idx * nx_, nx_) = model_->Ad(state, input);
      Bd_list_.middleCols(idx * nu_, nu_) = model_->Bd(state, input);

      residual_x_.segment(idx * nx_, nx_) = (model_->F_eval_ - state_next).cast<double>();

      if (nh_ > 0) {
        Cd_list_.middleCols(idx * nx_, nx_) = model_->Cd(state, input, params);
        Dd_list_.middleCols(idx * nu_, nu_) = model_->Dd(state, input, params);

        residual_h_.segment(idx * nh_, nh_) = (model_->h_eval_).cast<double>();


      }

    }

    if (nh_e_ > 0) {
      VectorAD state = x_guess_.segment(N_ * nx_, nx_);
      Vector params = parameters_.col(N_);

      Cd_e = model_->Cd_e(state, params);
      residual_h_.segment(N_ * nh_, nh_e_) = model_->he_eval_.cast<double>();
    }
  }

  void ComputeBounds() {
    lower_bound_ << (x_current_ - x_guess_.head(nx_)), residual_x_, -residual_h_;
    upper_bound_ << (x_current_ - x_guess_.head(nx_)), residual_x_, OsqpEigen::INFTY * Vector::Ones(nh_ * N_ + nh_e_);
  }

  void SetParametersAtStage(size_t stage, const Vector &parameters) {
    parameters_.col(stage) = parameters;
  }


  void SetXcurrent(const Vector &x_current) { x_current_ = x_current; }

  void SetUref(const Vector &u_ref) { cost_->u_ref_ = u_ref; }


  // Performs shifting repeating the last input
  void UpdateSolutionGuess() {
    x_guess_ += QPsolver_.getSolution().head(nx_ * (N_ + 1));
    u_guess_ += QPsolver_.getSolution().tail(nu_ * N_);

    x_guess_.head(nx_ * N_) = x_guess_.tail(nx_ * N_);
    u_guess_.head(nu_ * (N_ - 1)) = u_guess_.tail(nu_ * (N_ - 1));

    Vector last_x = x_guess_.tail(nx_);
    Vector last_u = u_guess_.tail(nu_);
    Vector next_guess = model_->integrator(last_x, last_u);

    x_guess_.tail(nx_) = next_guess;
  }

  bool SetupQPsolver() {
    // Settings
    //QPsolver_.settings()->SetVerbosity(false);
    QPsolver_.settings()->setWarmStart(true);
    QPsolver_.settings()->setPolish(true);
    QPsolver_.settings()->setMaxIteration(500);
    QPsolver_.settings()->setAbsoluteTolerance(0.001);
    QPsolver_.settings()->setRelativeTolerance(0.001);

    // Set the Initial data of the QP solver
    QPsolver_.data()->setNumberOfVariables(nx_ * (N_ + 1) + nu_ * N_);
    QPsolver_.data()->setNumberOfConstraints(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);
    QPsolver_.data()->setHessianMatrix(hessian_matrix_);
    QPsolver_.data()->setGradient(gradient_);
    QPsolver_.data()->setLinearConstraintsMatrix(constraint_matrix_);
    QPsolver_.data()->setLowerBound(lower_bound_);
    QPsolver_.data()->setUpperBound(upper_bound_);

    // instantiate the solver
    if (!QPsolver_.initSolver()) return 1;

    return 0;
  }

  void UpdateQPsolver() {
    QPsolver_.updateGradient(gradient_);
    QPsolver_.updateLinearConstraintsMatrix(constraint_matrix_);
    QPsolver_.updateBounds(lower_bound_, upper_bound_);
  }

  //void Preparation();

  //void Feedback(Vector &xcurrent);

  Vector SolveRTI(Vector &xcurrent) {

    SetXcurrent(xcurrent);

    ComputeSensitivitiesAndResiduals();
    ComputeCost();
    ComputeBounds();

    UpdateConstraintMatrix();
    UpdateHessian();
    UpdateGradient();

    UpdateQPsolver();
    QPsolver_.solve();

    Vector ctrl = u_guess_.head(nu_) + QPsolver_.getSolution().segment(nx_ * (N_ + 1), nu_);

    UpdateSolutionGuess();

    return ctrl;

  }




};

}