
#include <iostream>
#include <cmath>
#include <chrono>

#include "vurtis/vurtis.h"
#include "csvparser.hpp"

// Simple kinematic unicycle model with no constraints
class Model : public vurtis::ModelBase {
 public:
  explicit Model(double dt) : ModelBase(dt) {}

 private:
// Continuous-time dynamic model of the system for AD
  vurtis::VectorAD Dynamics(vurtis::VectorAD &state, vurtis::VectorAD &input) {

    vurtis::VectorAD state_dot{3};

    vurtis::real theta = state[2];

    vurtis::real v = input[0];
    vurtis::real omega = input[1];

    state_dot[0] = v * cos(theta);
    state_dot[1] = v * sin(theta);
    state_dot[2] = omega;

    return state_dot;
  }

  // dual-sided input constraints
  // constraints are of the type g(x,u)>=0
  vurtis::VectorAD Constraint(vurtis::VectorAD &state, vurtis::VectorAD &input, vurtis::Vector &params) {
    vurtis::VectorAD input_constraint(4);
    input_constraint[0] = input[0] + 1;
    input_constraint[1] = - input[0] + 1;
    input_constraint[2] = input[1] + 1;
    input_constraint[3] = - input[1] + 1;

    return input_constraint;
  }
  vurtis::VectorAD EndConstraint(vurtis::VectorAD &state, vurtis::Vector &params) {}

};

class Cost : public vurtis::CostBase {
 public:
  explicit Cost(vurtis::Vector& x_ref, vurtis::Vector& u_ref) : vurtis::CostBase(x_ref, u_ref) {}

  // function LeastSquareCost: R(x) such that CostFunctionPointwise=0.5*R(x)'*R(x)
  vurtis::VectorAD LeastSquareCost(vurtis::VectorAD &state, vurtis::VectorAD &input, vurtis::Vector &state_ref, vurtis::Vector &input_ref) {

    vurtis::VectorAD cost(4);

    cost[0] = 15 * (state[0]-state_ref[0]);
    cost[1] = 15 * (state[1]-state_ref[1]);
    cost[2] = std::sqrt(5) * (input[0]-input_ref[0]);
    cost[3] = std::sqrt(0.1) * (input[1]-input_ref[1]);

    return cost;
  }
};



int main() {

  std::ofstream state_trajectory;
  std::ofstream time_step;
  state_trajectory.open("state_trajectory.csv");
  time_step.open("time_step.csv");

  const double dT = 0.05; // sampling time (s)
  const size_t nx = 3; // state dimension
  const size_t nu = 2; // input dimension
  const size_t nz = 4; // cost dimension (number of elements of R s.t. cost = 0.5*R'R)
  const size_t nh = 4; // dimension of stagewise Constraint
  const size_t nh_e = 0; // dimension of end Constraint
  const size_t num_parameters = 0; // number of parameters

  const size_t N = 20; // length of control horizon (in steps)

  size_t Nsim = 600;

  std::chrono::steady_clock::time_point begin, end;
  double mean_time_temp = 0.0;
  double max_time = 1e-12;
  double elapsed_time = 0.0;


  vurtis::Vector ctrl;

  vurtis::Vector x_curr{nx};
  x_curr << 1.3, 0, M_PI / 2;

  // ------------------------------------------------------------------------------------------------------------------
  // Initialize reference
  vurtis::Vector x_ref{nx * (N + 1)};
  vurtis::Vector u_ref{nu * N};

  // Load data
  Eigen::MatrixXd desired_trajectory = load_csv<Eigen::MatrixXd>("data/trajectory_curve_20Hz.csv");

  // Set reference for initialization
  for (size_t ii = 0; ii <= N; ++ii) {
    x_ref.segment(ii * nx, nx) = desired_trajectory.row(ii);
  }

  vurtis::Vector u_r(nu);
  u_r << 0.2, 0;
  u_ref = u_r.replicate(N, 1);

  // ------------------------------------------------------------------------------------------------------------------

  auto unicycle_model = std::make_shared<Model>(dT);
  auto cost_function = std::make_shared<Cost>(x_ref, u_ref);
  vurtis::ProblemInit problem_init{nx, nu, nz, nh, nh_e, N, num_parameters, x_curr};
  vurtis::Solver solver(unicycle_model, cost_function, problem_init);


  for (size_t idx = 0; idx < Nsim; ++idx) {
    // Start clock for iteration timing statistics
    begin = std::chrono::steady_clock::now();

    for (size_t ii = 0; ii <= N; ++ii) {
      x_ref.segment(ii * nx, nx) = desired_trajectory.row(idx + ii);
    }

    // Update reference and solve the problem
    cost_function->x_ref_ = x_ref;
    ctrl = solver.SolveRTI(x_curr);
    //crtl = solver.GetControlInput();

    // Simulate next state
    x_curr = unicycle_model->integrator(x_curr, ctrl);


    // Iteration timing
    end = std::chrono::steady_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
        / 1000.0; // divided to get milliseconds with microsecond accuracy
    std::cout << "iteration time duration = " << elapsed_time << " [ms]" << std::endl;
    mean_time_temp += elapsed_time;
    max_time = std::max(elapsed_time, max_time);

    // Log data to file
    state_trajectory << x_curr[0] <<","<<x_curr[1] <<","<<x_curr[2]<<","<<ctrl[0]<<","<<ctrl[1] <<"\n";
    time_step << elapsed_time << "\n";

  }

  double mean_time = mean_time_temp / Nsim;
  std::cout << "mean time for one iteration =" << mean_time << " [ms]" << std::endl;
  std::cout << "maximum time for one iteration =" << max_time << " [ms]" << std::endl;

  state_trajectory.close();
  time_step.close();

  return 0;
}