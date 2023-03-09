
#include <iostream>
#include <cmath>
#include <chrono>

#include "vurtis/vurtis.h"
#include "csvparser.h"

// Simple kinematic unicycle model with input constraints
class Model : private vurtis::ModelBase {
 public:

    explicit Model(const double dt, const Eigen::VectorXd & params) : ModelBase(dt), nominal_params_(params) {}

    Eigen::VectorXd nominal_params_;


    vurtis::VectorAD DynamicsParams(vurtis::VectorAD &state, vurtis::VectorAD &input, vurtis::VectorAD &params) {
      vurtis::VectorAD state_dot(3);
      vurtis::real theta = state[2];
      vurtis::real R = params(0);
      vurtis::real D = params(1);

      vurtis::real wr = input[0];
      vurtis::real wl = input[1];

      state_dot[0] = (wr+wl)*R/2 * cos(theta);
      state_dot[1] = (wr+wl)*R/2 * sin(theta);
      state_dot[2] = (wr-wl)*R/D;

      return state_dot;
    }

    vurtis::Vector integrator(const vurtis::Vector &state_,const vurtis::Vector &input_) {
      vurtis::VectorAD state = state_;
      vurtis::VectorAD input = input_;

      vurtis::VectorAD state_next = step(state, input);

      int ns = 1;
      double h = dt_ / ns;

      for (int i = 0; i < ns; ++i) {
        vurtis::VectorAD k1 = Dynamics(state, input);

        vurtis::VectorAD temp1 = state + 0.5 * k1 * h;
        vurtis::VectorAD k2 = Dynamics(temp1, input);

        vurtis::VectorAD temp2 = state + 0.5 * k2 * h;
        vurtis::VectorAD k3 = Dynamics(temp2, input);

        vurtis::VectorAD temp3 = state + k3 * h;
        vurtis::VectorAD k4 = Dynamics(temp3, input);

        state_next += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
      }

      return state_next.cast<double>();

    }

 private:
    // Continuous-time dynamic model of the system for AD
    vurtis::VectorAD Dynamics(vurtis::VectorAD &state, vurtis::VectorAD &input) {
      vurtis::VectorAD params(nominal_params_);
      return DynamicsParams(state, input, params);
  }

  // constraints are of the type g(x,u)>=0
  vurtis::VectorAD Constraint(vurtis::VectorAD &state, vurtis::VectorAD &input, const vurtis::Vector &params) {
    // dual-sided input constraints
    vurtis::VectorAD input_constraint(4);
    input_constraint[0] = input[0] + 3;
    input_constraint[1] = - input[0] + 3;
    input_constraint[2] = input[1] + 3;
    input_constraint[3] = - input[1] + 3;

    return input_constraint;
  }
  vurtis::VectorAD EndConstraint(vurtis::VectorAD &state, const vurtis::Vector &params) {}

};

class Cost : public vurtis::CostBase {
 public:
  explicit Cost(const vurtis::Vector& x_ref, const vurtis::Vector& u_ref) : vurtis::CostBase(x_ref, u_ref) {}

  // function LeastSquareCost: R(x) such that CostFunctionPointwise=0.5*R(x)'*R(x)
  vurtis::VectorAD LeastSquareCost(vurtis::VectorAD &state, vurtis::VectorAD &input, const vurtis::Vector &state_ref, const vurtis::Vector &input_ref) {

    vurtis::VectorAD cost(5);

    cost[0] = 0 * (state[0]-state_ref[0]);
    cost[1] = 0 * (state[1]-state_ref[1]);
    cost[2] = 0 * (state[1]-state_ref[1]);
    cost[3] = 0.1 * (input[0]-input_ref[0]);
    cost[4] = 0.1 * (input[1]-input_ref[1]);

    return cost;
  }

    vurtis::VectorAD LeastSquareCostTerminal(vurtis::VectorAD &state, const vurtis::Vector &state_ref) {

      return We*(state-state_ref);
    }

  private:

    const vurtis::Matrix We = Eigen::Vector3d(10,10,0).asDiagonal();
};



int main() {

  std::ofstream state_trajectory;
  std::ofstream time_step;
  state_trajectory.open("data/state_trajectory.csv");
  time_step.open("data/time_step.csv");
  // Load data
  //Eigen::MatrixXd desired_trajectory = load_csv<Eigen::MatrixXd>("data/trajectory_curve_20Hz.csv");


  const double dT = 0.02; // sampling time (s)
  const int nx = 3; // state dimension
  const int nu = 2; // input dimension
  const int nz = 5; // cost dimension (number of elements of R s.t. cost = 0.5*R'R)
  const int nh = 4; // dimension of stagewise Constraint
  const int nh_e = 0; // dimension of end Constraint
  const int num_parameters = 0; // number of parameters

  const int N = 50; // length of control horizon (in steps)

  vurtis::Matrix parameters = vurtis::Matrix::Zero(num_parameters, N+1); // init parameters

  int Nsim = 2000;

  std::chrono::steady_clock::time_point begin, end;
  double mean_time_temp = 0.0;
  double max_time = 1e-12;
  double elapsed_time = 0.0;

  vurtis::Vector ctrl;

  vurtis::Vector x_curr(nx);
  x_curr << 0, 0, 0;

  vurtis::Vector x_des(nx);
  x_des << 1.5, 1.0, 0;


  // ------------------------------------------------------------------------------------------------------------------
  // Initialize reference
  vurtis::Vector x_ref(nx * (N + 1));
  vurtis::Vector u_ref(nu * N);

  x_ref = x_des.replicate(N+1, 1);
  u_ref = Eigen::Vector2d(0.0,0.0).replicate(N, 1);
  // ------------------------------------------------------------------------------------------------------------------

  Eigen::Vector2d params(0.1, 0.2);

  auto unicycle_model = std::make_shared<Model>(dT, params);
  auto cost_function = std::make_shared<Cost>(x_ref, u_ref);
  vurtis::ProblemInit problem_init{nx, nu, nz, nh, nh_e, N, parameters, x_curr};
  vurtis::Solver solver(unicycle_model, cost_function, problem_init);


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

    // Preparation phase for next iteration
    solver.Preparation();


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