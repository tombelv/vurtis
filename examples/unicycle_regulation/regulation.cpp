#include <iostream>
#include <cmath>
#include <chrono>

#include "vurtis/vurtis.h"
#include "csvparser.h"

#include "unicycle_model.h"
#include "cost_function.h"

#include"solver_sensitivity.h"


int main() {

  std::ofstream state_trajectory, input_trajectory, time_step, sensitivity;

  state_trajectory.open("data/state_trajectory.csv");
  input_trajectory.open("data/input_trajectory.csv");
  time_step.open("data/time_step.csv");
  sensitivity.open("data/sensitivity.csv");


  const double dT = 0.05; // sampling time (s)
  const int nx = 3; // state dimension
  const int nu = 2; // input dimension
  const int nz = nx+nu; // cost dimension (number of elements of R s.t. cost = 0.5*R'R)
  const int nh = 4; // dimension of stagewise Constraint
  const int nh_e = 0; // dimension of end Constraint
  const int num_parameters = 0; // number of parameters

  const int N = 20; // length of control horizon (in steps)

  vurtis::Matrix parameters = vurtis::Matrix::Zero(num_parameters, N+1); // init parameters

  int Nsim = 20;

  std::chrono::steady_clock::time_point begin, end;
  double mean_time_temp = 0.0;
  double max_time = 1e-12;
  double elapsed_time = 0.0;

  vurtis::Vector ctrl;

  vurtis::Vector x_curr(nx);
  x_curr << 0, 0, 0.1;

  vurtis::Vector x_des(nx);
  x_des << 0.3, 0.3, x_curr(2);


  // ------------------------------------------------------------------------------------------------------------------
  // Initialize reference
  vurtis::Vector x_ref(nx * (N + 1));
  vurtis::Vector u_ref(nu * N);

  x_ref = x_des.replicate(N+1, 1);
  u_ref = Eigen::Vector2d(0.0,1.0).replicate(N, 1);
  // ------------------------------------------------------------------------------------------------------------------

  Eigen::Vector2d model_params(0.1, 0.2);

  auto unicycle_model = std::make_shared<Model>(dT, nx, nu, nh, nh_e, model_params);
  auto cost_function = std::make_shared<Cost>(x_ref, u_ref);
  vurtis::ProblemInit problem_init{nx, nu, nz, nh, nh_e, N, parameters, x_curr};
  vurtis::SolverParametric solver(unicycle_model, cost_function, problem_init);

  vurtis::Matrix parametric_sensitivity;

  for (int idx = 0; idx < Nsim; ++idx) {
    // Start clock for iteration timing statistics
    begin = std::chrono::steady_clock::now();

/*    for (int ii = 0; ii <= N; ++ii) {
      x_ref.segment(ii * nx, nx) = desired_trajectory.row(idx + ii);
    }*/

    // Update reference and solve the problem
//    cost_function->x_ref_ = x_ref;
    ctrl = solver.Feedback(x_curr);

    // Simulate next state
    x_curr = unicycle_model->integrator(x_curr, ctrl);

    solver.ComputeSensitivity(parametric_sensitivity);

    // Preparation phase for next iteration
    solver.Preparation();

    vurtis::Vector x_traj = solver.GetStateTrajectory();
    vurtis::Vector u_traj = solver.GetInputTrajectory();

    // Iteration timing
    end = std::chrono::steady_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
        / 1000.0; // divided to get milliseconds with microsecond accuracy
    std::cout << "iteration time duration = " << elapsed_time << " [ms]" << std::endl;
    mean_time_temp += elapsed_time;
    max_time = std::max(elapsed_time, max_time);

    // Log data to file
    state_trajectory << x_traj.transpose().format(CSVFormat) <<"\n";
    input_trajectory << u_traj.transpose().format(CSVFormat) <<"\n";
    sensitivity << parametric_sensitivity.topRows(nx * (N + 1)).transpose().format(CSVFormat) << "\n";
    time_step << elapsed_time << "\n";

  }

  double mean_time = mean_time_temp / Nsim;
  std::cout << "mean time for one iteration =" << mean_time << " [ms]" << std::endl;
  std::cout << "maximum time for one iteration =" << max_time << " [ms]" << std::endl;

  state_trajectory.close();
  time_step.close();

  return 0;
}