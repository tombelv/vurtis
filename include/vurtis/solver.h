#pragma once

#include "OsqpEigen/OsqpEigen.h"
#include "vurtis/cost_base.h"
#include "vurtis/model_base.h"
#include "vurtis/setup.h"
#include "utils.h"
#include "csvparser.h"


namespace vurtis {

    class Solver {
        friend OsqpEigen::Solver;

    private:

        int nx_;
        int nu_;
        int nz_;
        int nh_;
        int nh_e_;

        int N_;

        Matrix Ad_list_;
        Matrix Bd_list_;
        Matrix Cd_list_;
        Matrix Cd_e;
        Matrix Dd_list_;


        Vector x_guess_;
        Vector u_guess_;

        Vector dw_;

//        Matrix parameters_;

        Vector x_current_;

        // Cost function
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

    protected:

        OsqpEigen::Solver QPsolver_;

    public:

        const std::shared_ptr<ModelBase> model_;
        const std::shared_ptr<CostBase> cost_;

        Solver(const std::shared_ptr<ModelBase> & model, const std::shared_ptr<CostBase> & cost, const ProblemSetup &problemParams) : model_{model}, cost_{cost} {

          x_current_ = problemParams.x0;

          nx_ = problemParams.nx;
          nu_ = problemParams.nu;
          nz_ = problemParams.nz;
          nh_ = problemParams.nh;
          nh_e_ = problemParams.nh_e;

          N_ = problemParams.N;

//          parameters_ = problemParams.nlp_parameters;

          InitMatricesToZero();

          residual_x_.resize(nx_ * N_);
          residual_h_.resize(nh_ * N_ + nh_e_);

          hessian_matrix_.resize(nx_ * (N_ + 1) + nu_ * N_, nx_ * (N_ + 1) + nu_ * N_);
          constraint_matrix_.resize(nx_ * (N_ + 1) + nh_ * N_ + nh_e_, nx_ * (N_ + 1) + nu_ * N_);

          lower_bound_.resize(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);
          upper_bound_.resize(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);


          InitSolutionGuess();

          InitHessian();

          ComputeGradient();

          InitConstraintMatrix();

          UpdateBounds();

          SetupQPsolver();

        }

        void InitMatricesToZero() {

          Ad_list_ = Matrix::Zero(nx_, nx_ * N_);
          Bd_list_ = Matrix::Zero(nx_, nu_ * N_);
          Cd_list_ = Matrix::Zero(nh_, nx_ * N_);
          Cd_e = Matrix::Zero(nh_e_, nx_);
          Dd_list_ = Matrix::Zero(nh_, nu_ * N_);

          R_list_ = Matrix::Zero(nz_, N_+1);
          dRdx_list = Matrix::Zero(nz_, nx_ * (N_ + 1));
          dRdu_list = Matrix::Zero(nz_, nu_ * N_);

          gradient_ = Vector::Zero(nx_ * (N_ + 1) + nu_ * N_);

        }

        void InitHessian() {


          std::vector<Eigen::Triplet<double>> tripletList;

          ComputeCost();

          tripletList.reserve(nx_*nx_ * (N_ + 1) + nu_*nu_ * N_ + 2*nx_*nu_*N_);


          for (int i = 0; i <  N_; ++i) {
            int idx_x = i*nx_;
            int idx_u = i*nu_;
            Matrix H_dxdx = (dRdx_list.middleCols(idx_x,nx_)).transpose()*dRdx_list.middleCols(idx_x,nx_);
            Matrix H_dudu = (dRdu_list.middleCols(idx_u,nu_)).transpose()*dRdu_list.middleCols(idx_u,nu_);

            Matrix H_dxdu = (dRdx_list.middleCols(idx_x,nx_)).transpose()*dRdu_list.middleCols(idx_u,nu_);
            Matrix H_dudx = H_dxdu.transpose();


            // State derivative terms
            for (int row_idx = 0; row_idx < nx_; ++row_idx) {
              for (int col_idx = 0; col_idx < nx_; ++col_idx)
                tripletList.push_back(Eigen::Triplet<double>(idx_x + row_idx, idx_x + col_idx, H_dxdx(row_idx, col_idx)));
            }
            // Input derivative terms
            for (int row_idx = 0; row_idx < nu_; ++row_idx) {
              for (int col_idx = 0; col_idx < nu_; ++col_idx)
                tripletList.push_back(Eigen::Triplet<double>(idx_u + nx_*(N_+1) + row_idx, idx_u + nx_*(N_+1) + col_idx, H_dudu(row_idx, col_idx)));
            }

            /* Cross derivative terms (dxdu)
             * This is the upper triangular block */
            for (int row_idx = 0; row_idx < nx_; ++row_idx) {
              for (int col_idx = 0; col_idx < nu_; ++col_idx)
                tripletList.push_back(Eigen::Triplet<double>(idx_x + row_idx, idx_u + nx_*(N_+1) + col_idx, H_dxdu(row_idx, col_idx)));
            }
            // There is a lower triangular symmetric block for dudx
            for (int row_idx = 0; row_idx < nu_; ++row_idx) {
              for (int col_idx = 0; col_idx < nx_; ++col_idx)
                tripletList.push_back(Eigen::Triplet<double>(idx_u + row_idx + nx_*(N_+1), idx_x  + col_idx, H_dudx(row_idx, col_idx)));
            }

          }

          // Add the end cost_ block (dim: nx_)
          { int i = nx_ * N_;

            int idx_x = i*nx_;
            Matrix H_dxdx = (dRdx_list.middleCols(i,nx_)).transpose()*dRdx_list.middleCols(i,nx_);

            for (int row_idx = 0; row_idx < nx_; ++row_idx) {
              for (int col_idx = 0; col_idx < nx_; ++col_idx)
                tripletList.push_back(Eigen::Triplet<double>(i + row_idx, i + col_idx, H_dxdx(row_idx, col_idx)));
            }
          }


          hessian_matrix_.setFromTriplets(tripletList.begin(), tripletList.end());

          hessian_matrix_.makeCompressed();

        }


// Commented out because not used anymore,kept for the moment
/*  void InitGradient() {

    for(int i = 0; i < N_; ++i) {
      Matrix dR(nz_,nx_+nu_);
      dR.leftCols(nx_) = dRdx_list.middleCols(i*nx_,nx_);
      dR.rightCols(nu_) = dRdu_list.middleCols(i*nu_,nu_);
      Vector grad = dR.transpose()*R_list_.col(i);
      gradient_.segment(i * nx_, nx_) = grad.head(nx_);
      gradient_.segment(nx_*(N_+1) + i * nu_, nu_) = grad.tail(nu_);
    }
    // Terminal cost
    Matrix dR = dRdx_list.middleCols(N_*nx_,nx_);
    gradient_.segment(N_ * nx_, nx_) = dR.transpose()*R_list_.col(N_);

  }*/

        void InitConstraintMatrix() {
          std::vector<Eigen::Triplet<double>> tripletList;

          tripletList.reserve((nx_ + 1 + nh_) * (nx_ * N_) + (1 + nh_e_) * nx_ + (nx_ + nh_) * nu_ * N_);

          // Build identity block
          for (int i = 0; i < nx_ * (N_ + 1); ++i)
            tripletList.push_back(Eigen::Triplet<double>(i, i, 1.0));

          for (int block = 0; block < N_; ++block) {
            for (int col = block * nx_; col < (block + 1) * nx_; ++col) {

              for (int row = (block + 1) * nx_; row < (block + 2) * nx_; ++row)
                tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));

              for (int row = nx_ * (N_ + 1) + block * nh_; row < nx_ * (N_ + 1) + (block + 1) * nh_; ++row)
                tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));
            }
          }

          for (int block = 0; block < N_; ++block) {
            for (int col = nx_ * (N_ + 1) + block * nu_; col < nx_ * (N_ + 1) + (block + 1) * nu_; ++col) {

              for (int row = (block + 1) * nx_; row < (block + 2) * nx_; ++row)
                tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));

              for (int row = nx_ * (N_ + 1) + block * nh_; row < nx_ * (N_ + 1) + (block + 1) * nh_; ++row)
                tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));
            }
          }

          for (int col = nx_ * N_; col < nx_ * (N_ + 1); ++col) {
            for (int row = nx_ * (N_ + 1) + nh_ * N_; row < nx_ * (N_ + 1) + nh_ * N_ + nh_e_; ++row)
              tripletList.push_back(Eigen::Triplet<double>(row, col, 0.0));
          }

          constraint_matrix_.setFromTriplets(tripletList.begin(), tripletList.end());
          constraint_matrix_.makeCompressed();

          ComputeSensitivitiesAndResiduals();

          UpdateConstraintMatrix();

        }

        void InitSolutionGuess() {
          //x_guess_ = cost_->x_ref_;
          x_guess_ = x_current_.replicate(N_+1,1);
          u_guess_ = cost_->u_ref_;
        }

        void UpdateHessian() {

          Vector nonzero(nx_*nx_*(N_+1) + nu_*nu_*N_ + 2*nx_*nu_*N_);

          for (int i = 0; i <  N_; ++i) {
            int idx_x = i*nx_;
            int idx_u = i*nu_;
            Matrix H_dxdx = (dRdx_list.middleCols(idx_x,nx_)).transpose()*dRdx_list.middleCols(idx_x,nx_);
            Matrix H_dudx = (dRdu_list.middleCols(idx_u,nu_)).transpose()*dRdx_list.middleCols(idx_x,nx_);
            for (int ii = 0; ii < nx_; ++ii) {
              nonzero.segment((nx_ + nu_) * (idx_x + ii), nx_) = H_dxdx.col(ii);
              nonzero.segment((nx_ + nu_) * (idx_x + ii) + nx_, nu_) = H_dudx.col(ii);
            }
          }

          // For now the terminal cost is zero
          { int i = N_;
            Matrix H_dxdx = (dRdx_list.middleCols(i*nx_,nx_)).transpose()*dRdx_list.middleCols(i*nx_,nx_);
            for (int ii = 0; ii < nx_; ++ii)
              nonzero.segment((nx_+nu_)*i*nx_ + nx_*ii, nx_) = H_dxdx.col(ii);
          }



          const int input_start_coeff = nx_*nx_*(N_+1) + nx_*nu_*N_;

          for (int i = 0; i <  N_; ++i) {
            int idx_x = i*nx_;
            int idx_u = i*nu_;
            Matrix H_dudu = (dRdu_list.middleCols(idx_u,nu_)).transpose()*dRdu_list.middleCols(idx_u,nu_);
            Matrix H_dxdu = (dRdx_list.middleCols(idx_x,nx_)).transpose()*dRdu_list.middleCols(idx_u,nu_);
            for (int ii = 0; ii < nu_; ++ii) {
              nonzero.segment((nx_ + nu_) * (idx_u + ii) + input_start_coeff, nx_) = H_dxdu.col(ii);
              nonzero.segment((nx_ + nu_) * (idx_u + ii) + nx_ + input_start_coeff, nu_) = H_dudu.col(ii);
            }
          }

          hessian_matrix_.coeffs() = nonzero;



        }

        void ComputeGradient() {

          for(int i = 0; i < N_; ++i) {
            Matrix dR(nz_,nx_+nu_);
            dR << dRdx_list.middleCols(i*nx_,nx_), dRdu_list.middleCols(i*nu_,nu_);
            Vector grad = dR.transpose()*R_list_.col(i);
            gradient_.segment(i * nx_, nx_) = grad.head(nx_);
            gradient_.segment(nx_*(N_+1) + i * nu_, nu_) = grad.tail(nu_);
          }
          // Terminal cost
          Matrix dR = dRdx_list.middleCols(N_*nx_,nx_);
          gradient_.segment(N_ * nx_, nx_) = dR.transpose()*R_list_.col(N_);
        }



        void UpdateConstraintMatrix() {

          Vector nonzero((1 + nx_ + nh_) * (nx_ * N_) + (1 + nh_e_) * nx_ + (nx_ + nh_) * nu_ * N_);
          for (int i = 0; i < nx_ * (N_); ++i) {
            nonzero(i * (nx_ + 1 + nh_)) = 1;
            nonzero.segment(i * (nx_ + 1 + nh_) + 1, nx_) = -Ad_list_.col(i);
            nonzero.segment(i * (nx_ + 1 + nh_) + 1 + nx_, nh_) = Cd_list_.col(i);
          }

          //nonzero.segment((nx_+1+nh_)*(nx_*N_),nx_) << Vector::Ones(nx_);
          for (int i = 0; i < nx_; ++i)
            nonzero.segment(i * (1 + nh_e_) + (nx_ + 1 + nh_) * (nx_ * N_), 1 + nh_e_) << 1, Cd_e.col(i);

          int input_start_coeff = (nx_ + 1 + nh_) * (nx_ * N_) + (1 + nh_e_) * nx_;

          for (int i = 0; i < nu_ * N_; ++i) {
            nonzero.segment(i * (nx_ + nh_) + input_start_coeff, nx_) = -Bd_list_.col(i);
            nonzero.segment(i * (nx_ + nh_) + input_start_coeff + nx_, nh_) = Dd_list_.col(i);
          }

          constraint_matrix_.coeffs() = nonzero;

        }

        void ComputeCost() {
          for (int idx = 0; idx < N_; ++idx) {
            VectorAD state = x_guess_.segment(idx * nx_, nx_);
            VectorAD input = u_guess_.segment(idx * nu_, nu_);
            Vector state_ref = cost_->x_ref_.segment(idx * nx_, nx_);
            Vector input_ref = cost_->u_ref_.segment(idx * nu_, nu_);
//            Vector params = parameters_.col(idx);

            Matrix dR = cost_->GradientCost(state,input,state_ref,input_ref, idx);
            dRdx_list.middleCols(idx * nx_, nx_) = dR.leftCols(nx_);
            dRdu_list.middleCols(idx * nu_, nu_) = dR.rightCols(nu_);

            R_list_.col(idx) = cost_->cost_eval_.cast<double>();
          }

          VectorAD state = x_guess_.segment(N_ * nx_, nx_);
          Vector state_ref = cost_->x_ref_.segment(N_ * nx_, nx_);
//          Vector params = parameters_.col(N_);

          Matrix dR =  cost_->GradientCostTerminal(state,state_ref, N_);
          dRdx_list.block(0, N_ * nx_, dR.rows(), nx_) = dR;
          R_list_.block(0, N_, dR.rows(), 1) = cost_->cost_eval_term_.cast<double>();


        }

        void ComputeSensitivitiesAndResiduals() {
          for (int idx = 0; idx < N_; ++idx) {
            VectorAD state = x_guess_.segment(idx * nx_, nx_);
            VectorAD state_next = x_guess_.segment((idx + 1) * nx_, nx_);
            VectorAD input = u_guess_.segment(idx * nu_, nu_);
//            Vector params = parameters_.col(idx);

            Ad_list_.middleCols(idx * nx_, nx_) = model_->Ad(state, input);
            Bd_list_.middleCols(idx * nu_, nu_) = model_->Bd(state, input);

            residual_x_.segment(idx * nx_, nx_) = (model_->F_eval_ - state_next).cast<double>();

            if (nh_ > 0) {
              Cd_list_.middleCols(idx * nx_, nx_) = model_->Cd(state, input, idx);
              Dd_list_.middleCols(idx * nu_, nu_) = model_->Dd(state, input, idx);

              residual_h_.segment(idx * nh_, nh_) = (model_->h_eval_).cast<double>();
            }

          }

          if (nh_e_ > 0) {
            VectorAD state = x_guess_.segment(N_ * nx_, nx_);
//            Vector params = parameters_.col(N_);

            Cd_e = model_->Cd_e(state, N_);
            residual_h_.segment(N_ * nh_, nh_e_) = model_->he_eval_.cast<double>();
          }
        }

        void UpdateBounds() {
          lower_bound_ << (x_current_ - x_guess_.head(nx_)), residual_x_, -1e15*Vector::Ones(nh_ * N_ + nh_e_);
          upper_bound_ << (x_current_ - x_guess_.head(nx_)), residual_x_, -residual_h_;
        }

//        void SetParametersAtStage(int stage, const Vector &parameters) {
//          parameters_.col(stage) = parameters;
//        }


        void SetXcurrent(const Vector &x_current) { x_current_ = x_current; }



        // Performs shifting repeating the last input
        void UpdateSolutionGuess() {
          x_guess_ += dw_.head(nx_ * (N_ + 1));
          u_guess_ += dw_.tail(nu_ * N_);
        }

        void ShiftInitialization() {
          x_guess_.head(nx_ * N_) = x_guess_.tail(nx_ * N_);
          u_guess_.head(nu_ * (N_ - 1)) = u_guess_.tail(nu_ * (N_ - 1));

          Vector last_x = x_guess_.tail(nx_);
          Vector last_u = u_guess_.tail(nu_);
          Vector next_guess = model_->integrator(last_x, last_u);

          x_guess_.tail(nx_) = next_guess;
        }

        bool SetupQPsolver() {
          // Settings
          QPsolver_.settings()->setVerbosity(true);
          QPsolver_.settings()->setWarmStart(true);
          QPsolver_.settings()->setPolish(true);
          QPsolver_.settings()->setMaxIteration(5000);
          QPsolver_.settings()->setAbsoluteTolerance(1e-4);
          QPsolver_.settings()->setRelativeTolerance(1e-4);
          //QPsolver_.settings()->setLinearSystemSolver(1);

          // Set the Initial data of the QP solver
          QPsolver_.data()->setNumberOfVariables(nx_ * (N_ + 1) + nu_ * N_);
          QPsolver_.data()->setNumberOfConstraints(nx_ * (N_ + 1) + nh_ * N_ + nh_e_);
          QPsolver_.data()->setHessianMatrix(hessian_matrix_);
          QPsolver_.data()->setGradient(gradient_);
          QPsolver_.data()->setLinearConstraintsMatrix(constraint_matrix_);
          QPsolver_.data()->setLowerBound(lower_bound_);
          QPsolver_.data()->setUpperBound(upper_bound_);

          // instantiate the solver
          if (!QPsolver_.initSolver()) return true;

          return false;
        }

        void UpdateQPsolver() {
          QPsolver_.updateHessianMatrix(hessian_matrix_);
          QPsolver_.updateGradient(gradient_);
          QPsolver_.updateLinearConstraintsMatrix(constraint_matrix_);
          QPsolver_.updateBounds(lower_bound_, upper_bound_);
        }



        Vector SolveRTI(Vector &xcurrent) {

          ComputeSensitivitiesAndResiduals();
          ComputeCost();

          UpdateConstraintMatrix();
          UpdateHessian();
          ComputeGradient();
          SetXcurrent(xcurrent);
          UpdateBounds();

          UpdateQPsolver();
          QPsolver_.solveProblem();

          dw_ = QPsolver_.getSolution();

          Vector ctrl = u_guess_.head(nu_) + QPsolver_.getSolution().segment(nx_ * (N_ + 1), nu_);

          UpdateSolutionGuess();

          ShiftInitialization();

          return ctrl;

        }

        // An example division of preparation and feedback phase.
        void Preparation() {

          ShiftInitialization();

          ComputeSensitivitiesAndResiduals();
          ComputeCost();

          UpdateConstraintMatrix();
          UpdateHessian();
          ComputeGradient();
        };

        void PreparationNoShift() {

          ComputeSensitivitiesAndResiduals();
          ComputeCost();

          UpdateConstraintMatrix();
          UpdateHessian();
          ComputeGradient();
        };

        Vector Feedback(Vector &xcurrent) {

          SetXcurrent(xcurrent);
          UpdateBounds();

          UpdateQPsolver();
          QPsolver_.solveProblem();

          dw_ = QPsolver_.getSolution();

          Vector ctrl = u_guess_.head(nu_) + QPsolver_.getSolution().segment(nx_ * (N_ + 1), nu_);

          UpdateSolutionGuess();

          return ctrl;
        };

        Vector SolveSQP(Vector &xcurrent, const int N) {

          Vector ctrl;

          for(int i = 0; i<N; ++i) {

            ComputeSensitivitiesAndResiduals();
            ComputeCost();

            UpdateConstraintMatrix();
            UpdateHessian();
            ComputeGradient();
            SetXcurrent(xcurrent);
            UpdateBounds();

            UpdateQPsolver();
            QPsolver_.solveProblem();

            dw_ = QPsolver_.getSolution();

            ctrl = u_guess_.head(nu_) + QPsolver_.getSolution().segment(nx_ * (N_ + 1), nu_);

            UpdateSolutionGuess();
          }


/*          ComputeSensitivitiesAndResiduals();
          ComputeCost();

          UpdateConstraintMatrix();
          UpdateHessian();
          ComputeGradient();*/

          return ctrl;
        }

        int GetSolverStatus() const { return (int)QPsolver_.getStatus();}

        const Vector GetMultipliers() {
          return QPsolver_.getDualSolution();
        }


        void debug() {
          std::cout << "Hessian matrix:\n" << hessian_matrix_ << std::endl;
          std::cout << "Constraint matrix:\n" <<  constraint_matrix_ << std::endl;
        }


        Vector GetStateTrajectory() {return x_guess_;}
        Vector GetInputTrajectory() {return u_guess_;}

        int GetN() {return N_;}
        Vector GetSolutionDelta() {return dw_;}
        Vector GetCurrentState() {return x_current_;}

//        Matrix GetNlpParams() {return parameters_;}

        SparseMatrixEigen GetHessianMatrix() {return hessian_matrix_;}
        SparseMatrixEigen GetConstraintMatrix() {return constraint_matrix_;}



    };

}