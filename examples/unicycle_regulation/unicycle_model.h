//
// Created by tom on 21/03/23.
//

#ifndef EXAMPLE_UNICYCLE_MODEL_H
#define EXAMPLE_UNICYCLE_MODEL_H

#include "vurtis/vurtis.h"

// Simple kinematic unicycle model with input constraints
class Model : public vurtis::ModelBase {
public:

    explicit Model(const double dt, const int nx, const int nu, const int nh, const int nh_e, const Eigen::VectorXd & params)
        :ModelBase(dt, nx, nu, nh, nh_e), nominal_params_(params), np_(params.size()) {}

    const Eigen::VectorXd nominal_params_;
    const int np_;


    vurtis::VectorAD Dynamics(vurtis::VectorAD &state, vurtis::VectorAD &input, vurtis::VectorAD &params) {
      vurtis::VectorAD state_dot(nx_);
      vurtis::ScalarAD theta = state(2);
      vurtis::ScalarAD R = params(0);
      vurtis::ScalarAD D = params(1);

      vurtis::ScalarAD wr = input[0];
      vurtis::ScalarAD wl = input[1];

      state_dot[0] = (wr+wl)*R/2 * cos(theta);
      state_dot[1] = (wr+wl)*R/2 * sin(theta);
      state_dot[2] = (wr-wl)*R/D;

      return state_dot;
    }

    vurtis::MatrixAD A(vurtis::VectorAD &state, vurtis::VectorAD &input, vurtis::VectorAD &params) {
      vurtis::MatrixAD res(nx_,nx_);
      res.setZero();

      vurtis::ScalarAD theta = state(2);

      vurtis::ScalarAD R = params(0);
      vurtis::ScalarAD D = params(1);

      vurtis::ScalarAD wr = input[0];
      vurtis::ScalarAD wl = input[1];

      res(0,2) = -(wr+wl)*R/2*sin(theta);
      res(1,2) = (wr+wl)*R/2*cos(theta);

      return res;
    }

    vurtis::MatrixAD B(vurtis::VectorAD &state, vurtis::VectorAD &input, vurtis::VectorAD &params) {
      vurtis::MatrixAD res(nx_,nu_);
      res.setZero();

      vurtis::ScalarAD theta = state(2);

      vurtis::ScalarAD R = params(0);
      vurtis::ScalarAD D = params(1);


      res(0,0) = res(0,1) = R/2*cos(theta);
      res(1,0) = res(1,1) = R/2*sin(theta);
      res(2,0) = R/D;
      res(2,1) = - R/D;

      return res;
    }

    vurtis::MatrixAD dFdxu(vurtis::VectorAD &state, vurtis::VectorAD &input, vurtis::VectorAD &params) {
      vurtis::VectorAD state_next = state;

      vurtis::MatrixAD res(nx_, nx_+nu_);
      res.leftCols(nx_) = vurtis::MatrixAD::Identity(nx_, nx_);

      int ns = 1;
      double h = dt_ / ns;

      for (int i = 0; i < ns; ++i) {
        vurtis::VectorAD k1 = Dynamics(state, input, params);

        vurtis::MatrixAD dk1dxu(nx_, nx_+nu_);
        dk1dxu.leftCols(nx_) = A(state_next, input, params)*res.leftCols(nx_);
        dk1dxu.rightCols(nu_) = A(state_next, input, params)*res.rightCols(nu_) + B(state_next, input, params);

        vurtis::VectorAD temp1 = state + 0.5 * k1 * h;
        vurtis::VectorAD k2 = Dynamics(temp1, input, params);

        vurtis::MatrixAD dk2dxu(nx_, nx_+nu_);
        dk2dxu.leftCols(nx_) = A(temp1, input, params)*(res.leftCols(nx_)+0.5*h*dk1dxu.leftCols(nx_));
        dk2dxu.rightCols(nu_) = A(temp1, input, params)*(res.rightCols(nu_)+0.5*h*dk1dxu.rightCols(nu_)) + B(temp1, input, params);

        vurtis::VectorAD temp2 = state + 0.5 * k2 * h;
        vurtis::VectorAD k3 = Dynamics(temp2, input, params);

        vurtis::MatrixAD dk3dxu(nx_, nx_+nu_);
        dk3dxu.leftCols(nx_) = A(temp2, input, params)*(res.leftCols(nx_)+0.5*h*dk2dxu.leftCols(nx_));
        dk3dxu.rightCols(nu_) = A(temp2, input, params)*(res.rightCols(nu_)+0.5*h*dk2dxu.rightCols(nu_)) + B(temp2, input, params);

        vurtis::VectorAD temp3 = state + k3 * h;
        vurtis::VectorAD k4 = Dynamics(temp3, input, params);

        vurtis::MatrixAD dk4dxu(nx_, nx_+nu_);
        dk4dxu.leftCols(nx_) = A(temp3, input, params)*(res.leftCols(nx_)+h*dk3dxu.leftCols(nx_));
        dk4dxu.rightCols(nu_) = A(temp3, input, params)*(res.rightCols(nu_)+h*dk3dxu.rightCols(nu_)) + B(temp3, input, params);

        state_next += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
        res += (h / 6) * (dk1dxu + 2 * dk2dxu + 2 * dk3dxu + dk4dxu);
      }

      return res;
    }


/*    vurtis::Vector integrator(const vurtis::Vector &state_,const vurtis::Vector &input_) {
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

    }*/

    vurtis::Vector GetModelParams() const {
      return nominal_params_;
    }

private:
    // Continuous-time dynamic model of the system for AD
    vurtis::VectorAD Dynamics(vurtis::VectorAD &state, vurtis::VectorAD &input) {
      vurtis::VectorAD params(nominal_params_);
      return Dynamics(state, input, params);
    }

    // constraints are of the type g(x,u)>=0
    vurtis::VectorAD Constraint(vurtis::VectorAD &state, vurtis::VectorAD &input, const vurtis::Vector &params) {
      // dual-sided input constraints
      double ub = 3;
      double lb = -3;
      vurtis::ScalarAD wr = input(0);
      vurtis::ScalarAD wl = input(1);
      vurtis::VectorAD input_constraint(nh_);
      input_constraint[0] = wr - lb;
      input_constraint[1] = - wr + ub;
      input_constraint[2] = wl - lb;
      input_constraint[3] = - wl + ub;
/*      input_constraint[0] = - wr + ub;
      input_constraint[1] = - wl + ub;*/

      return input_constraint;
    }
    vurtis::VectorAD EndConstraint(vurtis::VectorAD &state, const vurtis::Vector &params) {}

};

#endif //EXAMPLE_UNICYCLE_MODEL_H
